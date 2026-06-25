#!/usr/bin/env python3
"""Attribute the live memory at the real-peak instant to model modules.

Walks every block alive at the real (no-frag) peak time and buckets its size by
the deepest model-code frame in python_stack (lm_head Linear, loss, FSDP param,
optimizer copy, ...).  This is the "which module owns the peak" answer.

REQUIRES the run to have been launched with MS_DEV_LAUNCH_BLOCKING=1, otherwise
python_stack is truncated and everything lands in "(no model frame)".

Usage:
  python3 attribute_peak.py rank_2/memory_block.csv            # auto: real-peak time
  python3 attribute_peak.py rank_2/memory_block.csv --at 71367 # explicit instant
  python3 attribute_peak.py rank_2/memory_block.csv --top 20
"""
import argparse
import collections

from _memlib import (DEFAULT_MEMORY_POOL, compute_peaks, effective_end,
                     fmt_bytes, leaf_module, load_rows, max_observed_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--at", type=int, default=None, help="instant (default: real peak time)")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    peak_t = args.at if args.at is not None else compute_peaks(rows)["real"][0]
    max_obs = max_observed_time(rows)

    by_mod = collections.Counter()
    cnt = collections.Counter()
    total = nostack = 0
    for r in rows:
        try:
            start = int(float(r["start_time_stamp"]))
        except (TypeError, ValueError):
            continue
        if start == -1 or r.get("pool_type") != DEFAULT_MEMORY_POOL:
            continue
        if not (start <= peak_t < effective_end(r, max_obs)):
            continue
        size = int(float(r["size"]))
        total += size
        mod = leaf_module(r["python_stack"])
        if mod is None:
            nostack += size
            mod = "(no model frame / pool block)"
        by_mod[mod] += size
        cnt[mod] += 1

    print(f"{args.csv}  peak t={peak_t}  live total={fmt_bytes(total)}  "
          f"(no-model-frame={fmt_bytes(nostack)})")
    print("top modules holding live memory at peak:")
    for mod, s in by_mod.most_common(args.top):
        print(f"  {fmt_bytes(s):>11}  x{cnt[mod]:<4} {mod}")


if __name__ == "__main__":
    main()
