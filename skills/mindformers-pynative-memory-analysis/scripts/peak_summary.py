#!/usr/bin/env python3
"""Three-peak memory summary per rank, matching the davidfffan.github.io/tools view.

For each memory_block.csv prints:
  申请量峰值 (real,   no-frag)   : cumulative live size by real alloc/free
  理论峰值   (theory)            : cumulative live size by producer .. max(user) liveness
  内存池峰值 (actual, with-frag) : max actual_peak_memory
  real - theory (lifetime overhead / dead-but-held)
  actual - real (fragmentation)

A large real-theory gap => memory kept alive past its last logical use (see
block_lifetime.py to find the culprit).  A large actual-real gap => fragmentation.

Usage:
  python3 peak_summary.py rank_0/memory_block.csv [rank_1/memory_block.csv ...]
"""
import sys

from _memlib import compute_peaks, fmt_bytes, load_rows


def main(argv):
    if not argv:
        sys.exit(__doc__)
    for path in argv:
        p = compute_peaks(load_rows(path))
        r, t, a = p["real"], p["theory"], p["actual"]
        print(f"\n==== {path} ====")
        print(f"  申请量峰值(real,无碎片)  : {fmt_bytes(r[1]) if r else '-':>12}   @ t={r[0] if r else '-'}")
        print(f"  理论峰值  (theory)       : {fmt_bytes(t[1]) if t else '-':>12}   @ t={t[0] if t else '-'}")
        print(f"  内存池峰值(actual,含碎片): {fmt_bytes(a[1]) if a else '-':>12}   @ t={a[0] if a else '-'}")
        if r and t:
            print(f"  real - theory (生命周期冗余): {fmt_bytes(r[1] - t[1])}")
        if a and r:
            print(f"  actual - real (碎片)        : {fmt_bytes(a[1] - r[1])}")


if __name__ == "__main__":
    main(sys.argv[1:])
