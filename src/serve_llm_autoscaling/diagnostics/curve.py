"""Readable phase annotations for fine-grained session staircase curves."""


def phase_points(points):
    """Collapse a single up/hold/down staircase for annotation only.

    The workload and concurrency timeline retain every control point. Tiny
    paired transitions exist to compensate for AIPerf's nearest-integer
    interpolation; labeling each one obscures the three experimental phases.
    Other curves retain their original annotations.
    """
    if len(points) <= 4:
        return points
    changes = [(b["time_s"] - a["time_s"], b["sessions"] - a["sessions"])
               for a, b in zip(points, points[1:]) if a["sessions"] != b["sessions"]]
    if not changes or any(duration > .01 or abs(delta) != 1
                          for duration, delta in changes):
        return points
    peak = max(p["sessions"] for p in points)
    first = next(i for i, p in enumerate(points) if p["sessions"] == peak)
    last = max(i for i, p in enumerate(points) if p["sessions"] == peak)
    if first == 0 or last == len(points) - 1 or first == last:
        return points
    if any(b["sessions"] < a["sessions"] for a, b in zip(points[:first], points[1:first+1])):
        return points
    if any(b["sessions"] > a["sessions"] for a, b in zip(points[last:], points[last+1:])):
        return points
    # An explicit peak-hold endpoint precedes the first down-step's tiny
    # transition. It marks the nominal ramp-down start, before its first drop.
    hold_end = first + 1
    if (points[hold_end]["sessions"] != peak or
            points[hold_end]["time_s"] - points[first]["time_s"] < 60):
        return points
    return [points[0], points[first], points[hold_end], points[-1]]
