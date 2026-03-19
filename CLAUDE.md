# CLAUDE.md — MERaLiON2 vLLM Plugin

## Project overview

`vllm_plugin_meralion2` is a vLLM plugin that registers the MERaLiON-2-10B (Gemma2-based AudioLLM) architecture with vLLM's plugin system.

- **Package**: `vllm-plugin-meralion2` (`pyproject.toml`)
- **Entry point**: `vllm.general_plugins` → `vllm_plugin_meralion2:register`
- **Source root**: `src/vllm_plugin_meralion2/`
- **Tests**: `tests/`

## Key files

| File | Role |
|------|------|
| `src/vllm_plugin_meralion2/__init__.py` | `register()` — version gate, model registration, InputBatch patch |
| `src/vllm_plugin_meralion2/vllm0101.py` | Main adapter for vLLM 0.12.0 – 0.16.x |
| `src/vllm_plugin_meralion2/transformers_utils/no_repeat_logits_processor.py` | `NoRepeatNGramV1LogitsProcessor` (V1 entry-point) |
| `src/vllm_plugin_meralion2/transformers_utils/text_normalizer/` | Audiobench-compatible text normalizers for WER evaluation |
| `tests/test_asr_transcription_eval.py` | ASR WER integration tests (full dataset, needs live server) |
| `scripts/examples/serve_meralion2_ctm.sh` | Serve script for ASR/latency test modes |
| `scripts/compatibility/run_version_matrix_local.py` | Multi-version compatibility matrix runner |
| `scripts/compatibility/version_matrix_candidates.yaml` | Matrix config (vLLM x transformers versions) |

## Supported vLLM range

**`>=0.12.0, <0.17.0`** — tested on 0.12.0, 0.13.0, 0.14.0, 0.15.0, 0.15.1, 0.16.0.

All versions use `repetition_penalty=1.0` with the `InputBatch` monkey-patch that fixes
the V1 entry-point logits processor bug (see below).

### InputBatch monkey-patch (critical)

vLLM V1 sets `logitsprocs_need_output_token_ids=bool(custom_logitsprocs)` — ignoring
entry-point plugins. When `rep_penalty=1.0` (`no_penalties=True`), output token IDs are
`-1` placeholders, silently disabling `NoRepeatNGramV1LogitsProcessor`. The patch in
`__init__.py:_patch_logitsprocs_output_token_tracking()` forces the flag when
non-argmax-invariant entry-point processors are loaded. Safe — enables an existing
vLLM code path (async output-token-repair) with no side effects.

## ASR WER thresholds (Audiobench normalizers, full dataset)

| Dataset | N | WER threshold | Reference WER |
|---------|---|---------------|---------------|
| idpc_short_ASR_v2 | 122 | 0.16 | — |
| ste_test3 | 6 | 0.15 | — |
| ytb_asr_batch1 | 384 | 0.11 | 8.0% |
| ytb_asr_batch2 | 473 | 0.12 | 10.7% |
| ytb_asr_batch3_chinese | 206 | 0.17 | 13.1% |
| ytb_asr_batch3_malay | 200 | 0.18 | 15.7% |
| ytb_asr_batch3_tamil_v2 | 184 | 0.35 | 31.6% |

Chinese/Tamil/Malay WER is ~2-3pp above reference due to fixed 30s chunking vs
gateway's VAD-based chunking.

Tamil dataset: `ytb_asr_batch3_tamil_v2` → path `ytb_asr_batch3_tamil_filtered`.

## Attention backend

- vLLM `< 0.13.0`: `VLLM_ATTENTION_BACKEND=FLASHINFER` (env var only)
- vLLM `>= 0.13.0`: `--attention-backend FLASHINFER` CLI flag; unset env var in 0.13.0
- `serve_meralion2_ctm.sh` handles both automatically

## Running tests

### Serve

```bash
source .venvs/matrix/vllm_<version>__tf_4_57_6/bin/activate
bash scripts/examples/serve_meralion2_ctm.sh
```

### ASR WER test (full dataset, default)

```bash
export VLLM_TEST_BASE_URL=http://localhost:8063/v1
pytest tests/test_asr_transcription_eval.py -v -s -k "test_served_audiollm_asr_wer_lt_one"
```

For quick subset: `export ASR_TEST_MAX_SAMPLES=16`

### Full version matrix

```bash
python3 scripts/compatibility/run_version_matrix_local.py \
    --only-vllm 0.12.0 0.13.0 0.14.0 0.15.0 0.15.1 0.16.0 \
    --only-transformers 4.57.6
```

## Key serving parameters (GPU 1, TP=1, H100 79 GiB)

- `--gpu-memory-utilization 0.45` (0.5 causes OOM from audio encoder activation)
- `--max-num-seqs 12`
- `--tensor-parallel-size 1`
- `CUDA_VISIBLE_DEVICES=1`

## Venv naming convention

`.venvs/matrix/vllm_<VERSION>__tf_<TF_VERSION>` (dots → underscores)

## Model paths

- `/workspace/MERaLiON_local/MERaLiON-CTM-1512` — primary test model (Docker)
- `/home/yingxu/private_data/` — private ASR evaluation datasets

## CUDA / Docker notes

- `libcuda.so.1` symlink reverts to wrong version (570.x) on container restart;
  fix: `ln -sf libcuda.so.560.35.05 /usr/lib/x86_64-linux-gnu/libcuda.so.1`
- NVML "Failed to initialize" + `/dev/nvidia*` "Operation not permitted":
  NVIDIA cgroup reset — exit and restart docker
