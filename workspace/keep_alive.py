"""Keep an Anyscale workspace from idle-terminating.

Anyscale counts running Ray tasks as activity, so this script keeps a
short Ray task running back to back.

Usage:
    nohup python keep_alive.py > keep_alive.log 2>&1 &
    # stop it:
    pkill -f keep_alive.py
"""

import argparse
import os
import time
from datetime import datetime

import ray


@ray.remote(num_cpus=0)
def heartbeat(seconds: int) -> str:
    time.sleep(seconds)
    return datetime.now().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-seconds", type=int, default=60,
                        help="How long each heartbeat task runs.")
    parser.add_argument("--hours", type=float, default=0,
                        help="Stop after this many hours (0 = run forever).")
    args = parser.parse_args()

    # Anyscale workspaces set RAY_RUNTIME_ENV_HOOK, which snapshots the whole
    # workspace dir as the job's working_dir. The heartbeat needs no code, and
    # the snapshot can exceed Ray's 512 MiB package limit, so skip the hook.
    os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    deadline = time.time() + args.hours * 3600 if args.hours else None

    while deadline is None or time.time() < deadline:
        try:
            ts = ray.get(heartbeat.remote(args.task_seconds))
            print(f"[{ts}] heartbeat", flush=True)
        except Exception as e:  # keep going if Ray hiccups
            print(f"heartbeat failed: {e}", flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()