---
name: mindformers-pynative-memory-analysis
description: How to capture and read MindSpore memory_tracker data from MindFormers pynative training and turn an HBM peak into a module-attributed, actionable cause. Covers enabling memory_tracker + MS_DEV_LAUNCH_BLOCKING for full python_stack, the rank_<id>/memory_block.csv schema, the three-peak model (申请量/real no-frag, 理论/theory, 内存池/actual with-frag) matching davidfffan.github.io/tools, attributing the peak instant to model modules (lm_head logits, FSDP weights, AdamW fp32 master, loss intermediates), and a dead-but-held lifetime analysis that distinguishes a retention/scheduling problem from real fragmentation. Ends with a peak-classification decision tree.
when_to_use: User asks why HBM is high / where the memory peak is / 显存峰值 / 峰值在哪 / 为什么 OOM / 占用爆了; user wants to compare 申请量峰值 vs 理论峰值 vs 内存池峰值 or asks what the davidfffan.github.io/tools "理论峰值" means; user wants to know which module/op owns the peak (lm_head/logits, FSDP all-gather weights, optimizer master copy, loss); user suspects memory is held too long (real >> theory) or fragmented (actual >> real); conversation mentions memory_tracker / memory_block.csv / actual_peak_memory / MS_ALLOC_CONF / MS_DEV_LAUNCH_BLOCKING; user worries activation memory scales with micro-batch count under pipeline parallel.
---

# MindFormers Pynative Memory Analysis

How to turn a MindSpore `memory_tracker` dump into "this many GB is held by *this
module*, and here's why it won't free." Companion to
[`mindformers-pynative-training-run`](../mindformers-pynative-training-run/SKILL.md)
(launching, log layout, background-run pattern) and
[`mindformers-pynative-perf-analysis`](../mindformers-pynative-perf-analysis/SKILL.md)
(time, not memory).

This skill is for **memory peaks**, not throughput. Two steps is enough — the
tracker records the whole run and dumps at exit.

---

## Scripts in this skill

Under `scripts/` next to this SKILL.md (globally installed:
`~/.claude/skills/mindformers-pynative-memory-analysis/scripts/`). All read the
per-rank `memory_block.csv`; run them from the dir holding `rank_*/`, or pass a path.

| Script | Answers |
|---|---|
| [`peak_summary.py`](scripts/peak_summary.py) | The three peaks per rank + the two gaps. Start here. |
| [`attribute_peak.py`](scripts/attribute_peak.py) | Which **modules** own the live memory at the real-peak instant. Needs launch-blocking. |
| [`block_lifetime.py`](scripts/block_lifetime.py) | Per-block `start / last_used / freed / held_after` for a module or size filter — exposes dead-but-held retention. |
| [`_memlib.py`](scripts/_memlib.py) | Shared CSV parsing + the three-peak math (reused, don't run directly). |

---

## 1. Capture: two env vars

```bash
export MS_ALLOC_CONF="memory_tracker:True"   # dump rank_<id>/memory_block.csv at exit
export MS_DEV_LAUNCH_BLOCKING=1              # REQUIRED for module attribution (see below)
```

Then launch as usual (`--mode 1`, msrun for multi-card; see the training skill).
A `steps: 2` config is plenty. On exit each rank writes `./rank_<id>/`:

- `memory_block.csv` — one row per device memory block (what we analyse)
- `task.csv`, `tracker_graph.ir` — task list and op graph (rarely needed)

**Why `MS_DEV_LAUNCH_BLOCKING=1` is not optional here.** Pynative dispatches ops
asynchronously: the allocation runs on a runtime/device backend thread where the
user's Python frames are already gone, so the tracker captures only a degenerate
`traceback.py:...extract_stack` frame and `python_stack` is useless. Synchronous
execution runs the alloc inline on the dispatching thread, so `python_stack`
carries the real model call chain (`run_mindformer → trainer → scheduler →
gpt_model → transformer_block → ...`). It is slow — fine for a 2-step memory
probe, never for timing. (Equivalent: `ms.runtime.launch_blocking()` /
`context.pynative_synchronize=True` / yaml `context.pynative_synchronize: True`.)

**Watch for stale `rank_*` dirs.** They are written to CWD and not cleared between
runs; `rm -rf rank_*` before each launch or you'll read a previous config's dump.
With pipeline parallel, ranks split by stage — group them (e.g. pp=2, 8 cards:
ranks 0–3 = stage 0, 4–7 = stage 1) and analyse one rank per stage.

> Launch-flakiness note: on some hosts the first `msrun` after a prior run exits 1
> instantly (port/scheduler TIME_WAIT) with no logs — just retry on a fresh
> `--master_port`. Also: a foreground `sleep` is blocked by the harness; don't put
> `sleep` in a foreground Bash call.

---

## 2. The three peaks (what the davidfffan tool shows)

`memory_block.csv` columns we use: `start_time_stamp`, `end_time_stamp` (task-order
indices, **not** wall clock; a sentinel `2**63-1` means never-freed-in-trace),
`pool_type`, `size`, `actual_peak_memory`, `type`, `producer_task`, `node_name`,
`user_tasks`, `last_user_task`, `python_stack`.

`peak_summary.py` reproduces the davidfffan.github.io/tools computation exactly:

```bash
python3 scripts/peak_summary.py rank_2/memory_block.csv
```

| Peak | How it's computed | Meaning |
|---|---|---|
| **申请量峰值 (real, no-frag)** | cumulative live `size` by **real** `start..end`, default pool only | actual requested HBM, fragmentation excluded |
| **理论峰值 (theory)** | cumulative live `size` by **liveness** `producer_task .. max(user_tasks)` | ideal lower bound: each tensor lives only from produced to last-used, perfect reuse |
| **内存池峰值 (actual, with-frag)** | `max(actual_peak_memory)` column | real pool occupancy incl. fragmentation |

Ordering is always `theory ≤ real ≤ actual`. The two gaps are the whole diagnosis:

- **`real − theory` = lifetime overhead** — memory held alive *past its last logical
  use*. Theory frees a block at `max(user_tasks)`; reality frees it later. A big gap
  is NOT fragmentation — it's tensors kept around (retention / scheduling / no-consumer).
- **`actual − real` = fragmentation** — pool overhead. Usually the smaller gap.

---

## 3. Attribute the peak to a module

```bash
python3 scripts/attribute_peak.py rank_2/memory_block.csv      # auto: real-peak instant
```

Buckets every block alive at the peak instant by the deepest model frame in
`python_stack`. Typical owners and what they are:

| Module frame | What it is |
|---|---|
| `pynative/layers/linear.py:<n> construct` with `[seq, vocab]`-sized blocks | **lm_head logits** (output projection). Confirm: `size_bf16 / seq == vocab_size`. |
| `pynative/loss/loss.py:<n> forward/backward` | **cross-entropy intermediates** (full-vocab, often fp32 = 2× the bf16 logits) |
| `platform/mindspore/dtensor.py:194 set_data` | **FSDP all-gathered full weights** materialised on device |
| `platform/mindspore/dtensor.py:134 __copy__` ← `adamw.py _init_main_params` | **AdamW fp32 master weights** (`param.clone().float()`), persistent all run |
| `platform/mindspore/fully_shard/param.py:510 alloc_all_gather_outputs` | FSDP all-gather output buffers |
| `platform/mindspore/pipeline_parallel/backward.py:442` | pipeline backward saved grads |

Back-of-envelope to confirm a logit block: `bytes / dtype_size / seq_length`
should equal `vocab_size` (e.g. 1010 MB bf16 / 2 / 4096 = 129280). Only the output
head produces vocab-width tensors; attention/MLP activations are `hidden`/`inter`
wide and far smaller.

`seq_length` flips the dominant owner: long seq → lm_head logits + loss dominate;
very short seq → FSDP weights + optimizer master + grad buffers dominate.

---

## 4. Is it retention or fragmentation? — `block_lifetime.py`

When `real ≫ theory`, find what's held too long:

```bash
python3 scripts/block_lifetime.py rank_2/memory_block.csv --module "layers/linear.py:114" --min-mb 500 --at <peak_t>
```

Per block it prints `start / last_used / freed / held_after` where
`held_after = freed − last_used` is the **dead-but-held** span. A block last used
right after creation but freed much later is being pinned by a lingering reference,
not by computation.

**Case study this cracked (last-stage logits under 1F1B):**
- `held_after` ≈ one full training step, identical for every micro-batch.
- Blocks alive at peak == **micro-batch count** (8 micro → 8 live logits).
- Conclusion: activation memory scales with `num_microbatches` (GPipe-like), *not*
  with pipeline warmup depth (true 1F1B). Root cause: the pynative grad tape is
  **step-scoped** — all micro-batch forwards record into one
  `_pynative_executor` tape cleared only when the whole step's grad scope exits, so
  per-micro backward can't release its forward activations. Two holders confirmed:
  `fwd_outputs_cache[micro]` (released next step) and the shared executor tape
  (released after all backwards). **Implication: raising `micro_batch_num` raises
  last-stage peak linearly → OOM.** `chunk_loss_num` / `enable_loss_parallel` only
  shrink each block, they don't stop the pile-up; the real fix is a per-micro grad
  scope or activation recompute on the head.

This is also how you separate a real fix from a fake one: re-run, re-check
`real`/`theory` and `held_after`. A change that only moves `freed` earlier but not
before the peak instant does nothing for the peak (and loss must stay bit-identical
— see the perf skill's `compare_loss.py`).

---

## 5. Peak-classification decision tree

```
peak_summary.py
├─ actual − real large (fragmentation dominant)
│     → pool/allocator issue: tune mempool block size, reduce alloc churn. Rarely the main lever.
│
├─ real − theory large (lifetime overhead dominant)   ← most common, most fixable
│     attribute_peak.py → which module owns it?
│     ├─ lm_head logits / loss.py  → fused/chunked cross-entropy (chunk_loss_num>1);
│     │                              enable_loss_parallel ONLY helps with tp>1 (vocab-shard);
│     │                              block_lifetime.py → if count==micro count, it's the
│     │                              step-scoped grad tape (needs per-micro scope / recompute)
│     ├─ user_tasks empty + held   → no-consumer tensor: leak / late release; inspect refs
│     └─ recompute-able activations→ turn on / widen activation recompute
│
└─ real ≈ theory, both high (genuinely needed live memory)
      → not a release bug. Lower it structurally: more TP/EP/PP sharding,
        smaller batch/seq, optimizer-state dtype (e.g. optim_state_dtype: bf16),
        rebalance pipeline_parallel_layers_per_stage if one stage dominates.
```

---

## What this skill does NOT cover

- Time / throughput / overlap — that's `mindformers-pynative-perf-analysis`.
- Launching, card count, parallel-dim setup — `mindformers-pynative-training-run`.
- Implementing the per-micro grad-scope fix or fused linear-CE — those are code
  changes in `hyper_parallel` / `mindformers/pynative`, out of scope for analysis.
- Graph mode (`--mode 0`) — different allocator path.
