"""Shared helpers for parsing MindSpore memory_tracker ``memory_block.csv``.

A run launched with ``MS_ALLOC_CONF="memory_tracker:True"`` dumps, per rank,
``rank_<id>/memory_block.csv`` (+ task.csv, tracker_graph.ir).  Each row is one
device memory block with these columns (the ones we use):

  start_time_stamp  end_time_stamp  pool_type  size  actual_peak_memory
  type  producer_task  node_name  user_tasks  last_user_task  python_stack

``start/end_time_stamp`` are task-order indices (NOT wall clock); a block is
"alive" on ``[start, effective_end)``.  ``end_time_stamp`` may be a sentinel
(2**63-1) when the block was never freed within the trace.

This module reproduces, in Python, the three-peak computation used by the
davidfffan.github.io/tools memory visualiser so results match that tool:

  real  (申请量峰值 / no-frag) : cumulative live size by real start/end, default pool only
  theory(理论峰值)            : cumulative live size by producer_task .. max(user_tasks)
  actual(内存池峰值 / w/ frag) : max of the actual_peak_memory column

``real - theory`` = lifetime overhead (dead-but-held memory: blocks kept past
their last logical use).  ``actual - real`` = pool fragmentation.
"""
import csv
import re

DEFAULT_MEMORY_POOL = "DefaultEnhancedAscendMemoryPool"
END_SENTINEL_THRESHOLD = 9e18


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def max_observed_time(rows):
    m = 0
    for r in rows:
        s = _f(r.get("start_time_stamp"))
        if s is not None and s >= 0 and s > m:
            m = s
        e = _f(r.get("end_time_stamp"))
        if e is not None and 0 <= e < END_SENTINEL_THRESHOLD and e > m:
            m = e
    return m


def effective_end(r, max_obs):
    e = _f(r.get("end_time_stamp"))
    if e is not None and 0 <= e < END_SENTINEL_THRESHOLD:
        return int(e)
    return int(max_obs) + 1


def extract_users(s):
    return [int(x) for x in re.findall(r"\d+", s or "")]


def _peak_cumulative(changes):
    cur, peak = 0, None
    for t in sorted(changes):
        cur += changes[t]
        if peak is None or cur > peak[1]:
            peak = (t, cur)
    return peak


def compute_peaks(rows):
    """Return dict with ('time','value') tuples for real/theory/actual peaks."""
    max_obs = max_observed_time(rows)
    real, theory, actual = {}, {}, {}

    def add(m, k, v):
        m[k] = m.get(k, 0) + v

    for r in rows:
        start = _f(r.get("start_time_stamp"))
        size = _f(r.get("size")) or 0
        ap = r.get("actual_peak_memory", "")
        if ap not in (None, "") and start is not None:
            actual[int(start)] = float(ap)
        if start is None or start == -1 or r.get("pool_type") != DEFAULT_MEMORY_POOL:
            continue
        eff = effective_end(r, max_obs)
        add(real, int(start), int(size))
        add(real, eff - 1, 0)
        add(real, eff, -int(size))

    for r in rows:
        producer = r.get("producer_task")
        if not producer:
            continue
        users = extract_users(r.get("user_tasks", ""))
        if not users:
            continue
        start = _f(r.get("start_time_stamp"))
        if r.get("type") == "SomasOutput" and start == -1:
            continue
        pid = int(_f(producer))
        mx = max(users)
        size = int(_f(r.get("size")) or 0)
        add(theory, pid, size)
        add(theory, mx, 0)
        add(theory, mx + 1, -size)

    actual_peak = None
    for t in sorted(actual):
        if actual_peak is None or actual[t] > actual_peak[1]:
            actual_peak = (t, actual[t])

    return {
        "real": _peak_cumulative(real),
        "theory": _peak_cumulative(theory),
        "actual": actual_peak,
    }


def leaf_module(stack, roots=("mindformers/pynative", "mindformers/models", "hyper_parallel")):
    """Deepest python_stack frame that is real model code (skips nn/cell.py wrappers).

    Needs MS_DEV_LAUNCH_BLOCKING=1 at capture time, otherwise python_stack is
    truncated to a single traceback.extract_stack frame and this returns None.
    """
    best = None
    for seg in (stack or "").split("|"):
        if not seg.strip():
            continue
        d = dict(kv.split(":", 1) for kv in seg.split(";") if ":" in kv)
        f = d.get("File", "")
        if "nn/cell.py" in f or "traceback" in f:
            continue
        if any(k in f for k in roots):
            short = f.split("mindformers/")[-1].split("hyper_parallel/")[-1]
            best = f"{short}:{d.get('Line', '')} {d.get('Function', '')}".strip()
    return best


def fmt_bytes(b):
    if b is None:
        return "-"
    v = float(b)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or u == "TB":
            return f"{v:.2f} {u}"
        v /= 1024
