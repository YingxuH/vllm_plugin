# NoRepeatNGram Silent Failure in vLLM V1 — Bug Resolution Log

**Status**: Resolved (commit `b41c733`)
**Affected versions**: All vLLM V1 versions (0.10.1+) when using entry-point logits processors with `repetition_penalty=1.0`. Within current plugin scope: vLLM 0.12.0–0.16.x
**Root cause**: vLLM V1 does not enable output token tracking for entry-point logits processors

---

## 1. Background

MERaLiON-2 is a Gemma2-based AudioLLM. The model's transcription quality depends on a
`NoRepeatNGramLogitsProcessor` (n=6) that bans token sequences already seen in the
generation, preventing the runaway repetition loops common in autoregressive ASR.

In vLLM V1, logits processors can be registered via the `vllm.logits_processors` entry
point in `pyproject.toml`. The processor is instantiated once when the engine starts and
called on every decode step via `apply(logits)`. It receives per-slot state updates
(additions, removals, batch moves) through `update_state(BatchUpdate)`.

```toml
# pyproject.toml
[project.entry-points."vllm.logits_processors"]
NoRepeatNGramLogitsProcessor = "...no_repeat_logits_processor:NoRepeatNGramV1LogitsProcessor"
```

## 2. Symptom

After adding vLLM 0.15.0 to the test matrix (which upgrades FlashInfer from 0.5.3 to
0.6.x), some marginal audio samples entered runaway greedy-decoding loops — the model
would repeat the same n-gram thousands of times despite the NoRepeatNGram processor being
registered. ASR WER on affected samples exceeded 100%.

The bug was **silent**: no errors, no warnings. The processor was loaded, `apply()` was
called every step, but it never banned any tokens.

## 3. Initial Misdiagnosis: FlashInfer Numerics

### Hypothesis

FlashInfer 0.6.x enables FA3 kernel auto-selection on SM90a (H100). FA3 uses a different
softmax accumulation order than FA2/FlashInfer 0.5.x, producing slightly different
attention outputs (within IEEE 754 bounds but not bitwise-identical). The hypothesis was
that these numerical differences pushed marginal samples past a repetition tipping point.

### Evidence that seemed to confirm it

- The bug only appeared after upgrading to vLLM 0.15.0 (FlashInfer 0.6.x).
- On vLLM 0.12.0–0.14.0 (FlashInfer 0.5.3), the same samples decoded correctly.
- Setting `repetition_penalty=1.05` (the model's `generation_config.json` default)
  "fixed" the runaway loops on 0.15.0.

### Action taken (commit `c4b88bb`)

Capped `max_supported_version` at `<0.15.0` to exclude FlashInfer 0.6.x. Created a
separate branch (`feat/vllm-0.15-support`) to investigate.

### Action taken (commit `da0108f`)

Raised `repetition_penalty` from 1.0 to 1.05 for vLLM 0.15.0+. This masked the bug
by enabling output token tracking as a side effect (see Section 4).

## 4. Root Cause Discovery

### The real bug: output token IDs are `-1` placeholders

Debugging the `NoRepeatNGramV1LogitsProcessor.apply()` method revealed that
`output_tok_ids` (the live reference received via `BatchUpdate.added`) contained only
`-1` values — never the actual generated tokens. The n-gram lookup always found an empty
history, so no tokens were ever banned.

### Why `-1`?

vLLM V1's `InputBatch` manages a boolean flag:

```python
# vllm/v1/worker/gpu_input_batch.py  (simplified)
self.logitsprocs_need_output_token_ids = bool(custom_logitsprocs)
```

`custom_logitsprocs` is the list of logits processors passed via the
`--logits-processor-pattern` CLI flag or per-request `logits_processors` parameter.
**Entry-point processors are NOT included** — they are loaded separately by
`LogitsProcessorManager` from the `vllm.logits_processors` entry-point group.

When `logitsprocs_need_output_token_ids` is `False`, vLLM skips copying decoded tokens
into the per-request `output_token_ids` array. The array retains its initialization
value: `-1`.

### Why did `repetition_penalty=1.05` mask the bug?

vLLM optimizes penalty computation. When `repetition_penalty == 1.0`, it sets
`no_penalties=True` and skips the repetition-penalty path entirely. As part of this
optimization, output token tracking is also skipped (since penalties don't need it).

When `repetition_penalty=1.05` (or any value ≠ 1.0), `no_penalties=False`, and vLLM
enables output token tracking for the penalty computation. This **accidentally** also
populates the `output_token_ids` that the entry-point logits processor reads.

```
rep_penalty=1.0  → no_penalties=True  → output_token_ids = [-1, -1, ...]  → NoRepeatNGram BROKEN
rep_penalty=1.05 → no_penalties=False → output_token_ids = [real tokens]  → NoRepeatNGram works (by accident)
```

### Why didn't this appear on vLLM 0.12.0–0.14.0?

It DID — the NoRepeatNGram processor was silently broken on ALL versions with
`rep_penalty=1.0`. But the FlashInfer 0.5.3 attention numerics happened to produce
outputs that didn't trigger long repetition loops on the test samples. The FlashInfer
0.6.x upgrade changed the marginal probability distribution just enough to push some
samples into repetition territory, revealing the already-broken processor.

In other words: the processor was never working, but the model was "lucky" on 0.5.3.

## 5. The Fix (commit `b41c733`)

### Monkey-patch `InputBatch.__init__`

The fix wraps `InputBatch.__init__` to force `logitsprocs_need_output_token_ids=True`
when entry-point processors report `is_argmax_invariant()=False`:

```python
def _patch_logitsprocs_output_token_tracking() -> None:
    from vllm.v1.worker.gpu_input_batch import InputBatch
    _orig_init = InputBatch.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if (hasattr(self, "logitsprocs")
                and self.logitsprocs.non_argmax_invariant
                and not self.logitsprocs_need_output_token_ids):
            self.logitsprocs_need_output_token_ids = True

    InputBatch.__init__ = _patched_init
```

### Why this is safe

The flag enables an **existing vLLM code path**: async output-token repair in the
scheduler. When `True`, the scheduler copies decoded tokens into the output array on
each step. This is the same path used when `repetition_penalty ≠ 1.0`. The only cost
is a small per-step memcpy that was already happening for most real-world deployments
(which typically use `rep_penalty > 1.0`).

### Result

- Reverted `repetition_penalty` back to `1.0` across ALL versions (0.12.0–0.16.x)
- All 6 vLLM versions pass full-dataset ASR evaluation with identical WER
- Unified version range: `>=0.12.0, <0.17.0` (no more per-version rep_penalty)

## 6. Key Lessons

### Red herring pattern: correlated ≠ causal

The FlashInfer upgrade and the bug appearing were correlated in time but not causally
linked. The real cause (broken output token tracking) existed on all versions. The
FlashInfer change merely exposed it by altering which samples hit repetition.

### Silent failures in ML pipelines are the most dangerous

The processor was loaded, called, received logits, and returned them — all without error.
The only observable symptom was degraded ASR quality on a subset of samples, which was
initially attributed to attention numerics rather than a completely non-functional
component.

### Side-effect coupling masks bugs

`repetition_penalty=1.05` "fixed" the bug not because penalty=1.05 is better for the
model, but because it triggered an unrelated code path (penalty token tracking) that
happened to populate the array the logits processor needed. This kind of accidental
coupling makes bugs extremely hard to diagnose.

### Debugging approach that worked

1. Add logging inside `apply()` to print `output_tok_ids` — immediately revealed `-1`s
2. Trace backward: who populates `output_tok_ids`? → `InputBatch`
3. Read `InputBatch.__init__` source: flag set from `custom_logitsprocs` only
4. Confirm entry-point processors are not in `custom_logitsprocs`
5. Test: force flag to `True` → processor works on all versions

## 7. Files Involved

| File | Role |
|------|------|
| `src/vllm_plugin_meralion2/__init__.py` | Monkey-patch `InputBatch.__init__` |
| `src/vllm_plugin_meralion2/transformers_utils/no_repeat_logits_processor.py` | `NoRepeatNGramV1LogitsProcessor` — the affected processor |
| `pyproject.toml` | Entry-point registration under `vllm.logits_processors` |
| `vllm/v1/worker/gpu_input_batch.py` (upstream vLLM) | Source of the bug: `logitsprocs_need_output_token_ids = bool(custom_logitsprocs)` |

## 8. Commit Timeline

| Commit | Date | Description |
|--------|------|-------------|
| `7559f8d` | — | Initial V1 entry-point support with `rep_penalty=1.0` |
| `c4b88bb` | Mar 17 | Cap at `<0.15.0` (FlashInfer misdiagnosis) |
| `da0108f` | Mar 17 | Raise `rep_penalty` to 1.05 for 0.15.0+ (masked fix) |
| `b41c733` | Mar 19 | **Real fix**: monkey-patch InputBatch output token tracking |
| `e600257` | Mar 19 | Align ASR evaluation with Audiobench normalizers (clean test run) |
