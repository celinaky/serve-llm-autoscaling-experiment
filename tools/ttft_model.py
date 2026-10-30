#!/usr/bin/env python3
"""Predict client TTFT from the engine load a request finds and its own size.

Fits one linear model per TTFT component to the per-request table that
tools/load_breakdown.py writes (L = uncached prefill token load the request
found on its engine at arrival, U = its own uncached tokens, I = its full
prompt length, all in K tokens; S = steps it needs alone; B = chunk budget):

  queue    = q0 + q1 L                      the load drains first come, first served
  prefill  = p0 + p1 U + p2 S + p3 min(L, B) S
                                            own work, plus each step slowing until
                                            the load fills the step budget
  before   = a0 + a1 I + a2 L               proxy, router, tokenizing the full prompt
  after    = b0 + b1 I                      detokenize, stream

The components sum to the predicted client TTFT, so the size of the request
is a continuous input rather than a cold/warm split.

usage: python tools/ttft_model.py runs/<run_dir>

Writes analysis/ttft_model.{png,json} in the run.
"""
import argparse, csv, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = [  # stacked bottom to top; categorical slots 1-4, as in load_breakdown
    ("prefill_ms", "Prefill steps (engine)", "#2a78d6"),
    ("queue_ms", "Queue (engine)", "#eb6834"),
    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
]
# Request sizes are ordered magnitudes: one blue ramp, light to dark.
SIZE_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# Bands of own uncached tokens (K) for the observed points; the curve for each
# band is drawn at the band's median request.
BANDS = [(0, 4), (4, 16), (16, 48), (48, math.inf)]
LOAD_BINS = [(0, 0), (0, 16), (16, 32), (32, 64), (64, math.inf)]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

ap = argparse.ArgumentParser()
ap.add_argument("run")
a = ap.parse_args()
R = a.run.rstrip("/")

rows = list(csv.DictReader(open(f"{R}/analysis/load_requests.csv")))
col = lambda k: np.array([float(r[k]) for r in rows])
L, U, I = col("engine_load_tokens") / 1000, col("own_uncached") / 1000, col("isl") / 1000
S = col("min_steps")
B = float(rows[0]["chunk_budget"]) / 1000


def features(name, L, U, I, S):
    one = np.ones_like(L)
    return {
        "queue_ms": [one, L],
        "prefill_ms": [one, U, S, np.minimum(L, B) * S],
        "pre_ms": [one, I, L],
        "post_ms": [one, I],
    }[name]


TERMS = {"queue_ms": ["1", "L"], "prefill_ms": ["1", "U", "S", "min(L,B)*S"],
         "pre_ms": ["1", "I", "L"], "post_ms": ["1", "I"]}
model = {}
for key, *_ in COMPONENTS:
    X = np.column_stack(features(key, L, U, I, S))
    y = col(key)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    model[key] = {"terms": TERMS[key], "coef_ms": coef.tolist(), "r2": r2}


def predict(key, L, U, I, S):
    return np.column_stack(features(key, L, U, I, S)) @ np.array(model[key]["coef_ms"])


total = sum(predict(k, L, U, I, S) for k, *_ in COMPONENTS)
y = col("client_ttft_ms")
r2_total = 1 - ((y - total) ** 2).sum() / ((y - y.mean()) ** 2).sum()
json.dump({"units": "L, U, I in K tokens; coefficients in ms", "chunk_budget_k": B,
           "components": model, "client_ttft_r2": r2_total, "requests": len(rows)},
          open(f"{R}/analysis/ttft_model.json", "w"), indent=2)
for key, m in model.items():
    print(f"{key:11} R²={m['r2']:.3f}  " + "  ".join(f"{t}: {c:.1f}" for t, c in zip(m["terms"], m["coef_ms"])))
print(f"client TTFT R²={r2_total:.3f} over {len(rows)} requests")

# Each band's curve is drawn for its median request (uncached size, prompt
# length and steps needed alone), so the curves are real request shapes.
reps = []
for (lo, hi), color in zip(BANDS, SIZE_COLORS):
    m = (U >= lo) & (U < hi)
    i = np.argsort(U[m])[m.sum() // 2]
    reps.append({"lo": lo, "hi": hi, "color": color, "n": int(m.sum()), "mask": m,
                 "U": U[m][i], "I": I[m][i], "S": S[m][i]})

loads = sorted(L)
p95 = loads[int(len(loads) * 0.95)]
xmax = 96
grid = np.linspace(0, xmax, 200)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig = plt.figure(figsize=(14, 11), facecolor=SURFACE)
gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.42, wspace=0.12)
top = fig.add_subplot(gs[0, :])
bottoms = [fig.add_subplot(gs[1, i]) for i in range(4)]
for ax in bottoms[1:]:
    ax.sharey(bottoms[0])
X_LABEL = "uncached prefill token load the request found on its engine (K tokens)"


def band_label(rep):
    hi = "+" if rep["hi"] == math.inf else f"–{rep['hi']:.0f}K"
    return f"{rep['lo']:.0f}K{hi} uncached"


for rep in reps:
    shape = [np.full_like(grid, rep[k]) for k in ("U", "I", "S")]
    pred = sum(predict(k, grid, *shape) for k, *_ in COMPONENTS) / 1000
    top.plot(grid, pred, color=rep["color"], linewidth=2,
             label=f"{band_label(rep)} (curve at {rep['U']:.1f}K, n={rep['n']})")
    m = rep["mask"]
    for lo, hi in LOAD_BINS:
        sel = m & ((L == 0) if hi == 0 else ((L > lo) & (L <= hi)))
        if sel.sum() == 0:
            continue
        x = 0 if hi == 0 else min(L[sel].mean(), xmax)
        top.plot(x, y[sel].mean() / 1000, "o", color=rep["color"], markersize=8,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
top.axvline(p95, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
top.text(p95 + 1, 0.2, "→ top 5% of observed loads: few requests, mostly extrapolated", fontsize=8, color=INK2)
top.set_title(f"Predicted client TTFT (lines) and observed means per load bin (dots), "
              f"by the request's own uncached tokens. Model R² = {r2_total:.2f}",
              loc="left", fontsize=11, color=INK)
top.set_ylabel("seconds")
top.set_xlabel(X_LABEL)
top.set_xlim(0, xmax)
top.set_ylim(bottom=0)
top.legend(loc="upper left", frameon=False, fontsize=9)

for ax, rep in zip(bottoms, reps):
    shape = [np.full_like(grid, rep[k]) for k in ("U", "I", "S")]
    parts = [np.maximum(predict(k, grid, *shape), 0) / 1000 for k, *_ in COMPONENTS]
    ax.stackplot(grid, parts, colors=[c for *_, c in COMPONENTS],
                 labels=[lbl for _, lbl, _ in COMPONENTS], edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(p95, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_title(f"{band_label(rep)}: request of {rep['U']:.1f}K uncached,\n"
                 f"{rep['I']:.0f}K prompt, {rep['S']:.0f} step{'s' if rep['S'] != 1 else ''} alone",
                 loc="left", fontsize=9.5, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("load found on its engine (K tokens)")
bottoms[0].set_ylabel("predicted seconds")
for ax in bottoms[1:]:
    ax.tick_params(labelleft=False)
handles, labels = bottoms[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.005), frameon=False, ncol=4, fontsize=9)

for ax in [top, *bottoms]:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(f"{os.path.basename(R)}: client TTFT predicted from the engine load a request "
             f"finds and its own uncached tokens", x=0.01, ha="left", fontsize=13)
fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.11)
fig.savefig(f"{R}/analysis/ttft_model.png", dpi=130, facecolor=SURFACE)
print(f"wrote {R}/analysis/ttft_model.png and .json")
