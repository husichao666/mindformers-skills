"""List the lifetime of blocks matching a module/size filter, exposing
"dead-but-held": how long a block stays allocated AFTER its last logical use.

This is the tool that distinguishes a true leak / retention bug from normal use.
For each matching block it prints:
  start        : allocation task index
  last_used    : last_user_task column (block is logically dead after this)
  freed        : effective end (sentinel-clamped); when the pool actually released it
  held_after   : freed - last_used  (the dead-but-held span — the wasted lifetime)

Case study it cracked: last-stage lm_head logits show last_used right after their
own micro-batch backward, but held_after ~= one full training step, and the count
of simultaneously-live logits == micro-batch count.  That proved the pynative grad
tape is step-scoped (GPipe-like activation memory ~ N_micro), not 1F1B.

Usage:
  # filter by leaf module (substring of "<file>:<line>"), e.g. the lm_head Linear
  python3 block_lifetime.py rank_2/memory_block.csv --module "layers/linear.py:114"
  # filter by minimum size and show how many are alive at a given instant
  python3 block_lifetime.py rank_2/memory_block.csv --min-mb 500 --at 71367
"""
import argparse

from _memlib import (DEFAULT_MEMORY_POOL, effective_end, leaf_module,
                     load_rows, max_observed_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--module", default=None, help="substring of leaf '<file>:<line> <fn>'")
    ap.add_argument("--min-mb", type=float, default=0.0)
    ap.add_argument("--at", type=int, default=None, help="also report how many alive at this instant")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    max_obs = max_observed_time(rows)
    blocks = []
    for r in rows:
        if r.get("pool_type") != DEFAULT_MEMORY_POOL:
            continue
        try:
            start = int(float(r["start_time_stamp"]))
            size = int(float(r["size"]))
        except (TypeError, ValueError):
            continue
        if start == -1 or size < args.min_mb * 2 ** 20:
            continue
        mod = leaf_module(r["python_stack"]) or ""
        if args.module and args.module not in mod:
            continue
        eff = effective_end(r, max_obs)
        try:
            last_used = int(float(r.get("last_user_task")))
        except (TypeError, ValueError):
            last_used = start
        blocks.append((start, last_used, eff, size, mod))

    blocks.sort()
    print(f"{args.csv}: {len(blocks)} matching blocks "
          f"(module~{args.module!r}, >= {args.min_mb}MB)")
    print(f"{'start':>8} {'last_used':>9} {'freed':>8} {'held_after':>10} {'sizeMB':>7}  module")
    for s, lu, e, sz, mod in blocks:
        print(f"{s:>8} {lu:>9} {e:>8} {e - lu:>10} {sz / 2 ** 20:>7.0f}  {mod[:48]}")

    if args.at is not None:
        alive = [b for b in blocks if b[0] <= args.at < b[2]]
        live_gb = sum(b[3] for b in alive) / 2 ** 30
        print(f"\nalive @ t={args.at}: {len(alive)} blocks = {live_gb:.2f} GB")


if __name__ == "__main__":
    main()
