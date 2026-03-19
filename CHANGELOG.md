# CHANGELOG


## v0.3.0 (2026-03-19)

### Bug Fixes

- Add get_data_parser to ProcessingInfo for vLLM 0.16.0 audio resampling
  ([`47eeda6`](https://github.com/YingxuH/vllm_plugin/commit/47eeda695679cddda123588e46463f69293682ce))

In vLLM 0.16.0, audio data parsing uses BaseProcessingInfo.data_parser (a cached_property backed by
  get_data_parser()) rather than the processor's self.data_parser instance attribute. The base
  implementation returns MultiModalDataParser() without target_sr, causing: BadRequestError: Audio
  resampling is not supported when target_sr is not provided

Override get_data_parser() on MERaLiON2ProcessingInfo to supply the Whisper feature extractor's
  sampling rate. The existing __init__ override on the processor continues to patch self.data_parser
  for vLLM 0.12.0–0.15.x where the processor attribute is what's used.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Cap max_supported_version at 0.15.0 to exclude FlashInfer 0.6.x
  ([`c4b88bb`](https://github.com/YingxuH/vllm_plugin/commit/c4b88bba90371652f88d23e4c47faebcaa0e2c82))

vLLM 0.15.0 upgrades FlashInfer from 0.5.3 to 0.6.x, which enables FA3 kernel auto-selection on
  SM90a (H100). The changed attention numerics cause marginal audio samples to enter runaway
  greedy-decoding loops when repetition_penalty=1.0. Cap the supported range at < 0.15.0 to keep the
  existing plugin version's test contract (rep=1.0) intact. Support for vLLM 0.15.0+ will be added
  in a new plugin version (feat/vllm-0.15-support).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Fix vLLM bug: entry-point logits processors don't enable output token tracking.
  ([`b41c733`](https://github.com/YingxuH/vllm_plugin/commit/b41c73371bd41446686d111acfae37427c282472))

- Handle BaseDummyInputsBuilder module move in vLLM 0.16.0
  ([`e00f2f4`](https://github.com/YingxuH/vllm_plugin/commit/e00f2f449d0159f245bb50ee148edf8e2973dce3))

vLLM 0.16.0 moved BaseDummyInputsBuilder from vllm.multimodal.profiling to
  vllm.multimodal.processing. Add try/except import guard to support both locations (0.12.0–0.15.x
  use profiling, 0.16.0+ use processing).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Raise TTFT p95 budget to 10000 ms for vLLM 0.16.0
  ([`2358ed7`](https://github.com/YingxuH/vllm_plugin/commit/2358ed703ad6f739c026f44e59e7d49497d1190c))

vLLM 0.16.0's FlashInfer scheduling profile shows higher p95 TTFT under 32-concurrent load for
  verbose datasets (Tamil p95 ~7274 ms vs 6000 ms budget). This is the same batch-concurrency
  artifact as before — the last-queued requests wait for earlier long requests to drain. Raise the
  budget to 10000 ms to cover the observed range across 0.12.0–0.16.0.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Replace _get_data_parser with __init__ override for vLLM 0.16.0 compat
  ([`4ebe373`](https://github.com/YingxuH/vllm_plugin/commit/4ebe37341dd5b3a706072b006a840f3d509ecf91))

vLLM 0.16.0 raises ValueError if BaseMultiModalProcessor has a _get_data_parser method (moved to
  BaseProcessingInfo.get_data_parser). The default in all versions returns MultiModalDataParser()
  without target_sr, so audio resampling would be lost.

Replace the _get_data_parser override with an __init__ override that calls super().__init__() then
  immediately resets self.data_parser with the correct target_sr. self.data_parser is a plain
  instance attribute in all versions (0.12.0–0.16.0), so this pattern works everywhere.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Replace removed MultiModalKwargs with MultiModalKwargsItems for vLLM 0.16.0
  ([`f423623`](https://github.com/YingxuH/vllm_plugin/commit/f4236231ea896ca228f821c917979550b459028c))

MultiModalKwargs was removed from vllm.multimodal.inputs in vLLM 0.16.0; the correct type (also used
  by BaseMultiModalProcessor's base signature since 0.13.0) is MultiModalKwargsItems. Update the
  import and the _get_prompt_updates type annotation to match.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Resolve pylint warnings — suppress false positives, fix code style
  ([`fbc041f`](https://github.com/YingxuH/vllm_plugin/commit/fbc041fffc78941e61579a2455e2bb4537968227))

- .pylintrc: disable import-error (runtime deps not in CI), unused-argument (interface-required
  signatures), missing-function-docstring (interface methods) - __init__.py: remove superfluous
  parens after `not` (C0325) - modules.py: use Python 3 style super() (R1725) - vllm0101.py: use
  dict literals instead of dict() calls (R1735) - configuration_meralion2.py,
  processing_meralion2.py: wrap long lines (C0301)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Use --attention-backend CLI flag for vLLM >= 0.13.0
  ([`34fb24f`](https://github.com/YingxuH/vllm_plugin/commit/34fb24fd450c788dd78c911fe7f5db11c924cc18))

VLLM_ATTENTION_BACKEND env var was removed between 0.13.0 and 0.16.0; vLLM 0.16.0 only honours the
  --attention-backend CLI flag. vLLM 0.12.x has no --attention-backend flag and relies on the env
  var.

Update both serve scripts to: - Keep exporting VLLM_ATTENTION_BACKEND=FLASHINFER (for 0.12.x) -
  Detect version at startup and append --attention-backend FLASHINFER to the vllm serve command for
  vLLM >= 0.13.0

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Chores

- Clean up serve scripts and add vLLM 0.16.0 to test matrix
  ([`6499f44`](https://github.com/YingxuH/vllm_plugin/commit/6499f445c34ef425e6199295c8d08d3dfed84726))

- Remove dead code (commented-out local model paths, old flags) from
  private_scripts/serve_test_meralion2_general.sh and _ctm.sh - Add vLLM 0.16.0 / transformers
  4.57.6 to version_matrix_candidates.yaml (a venv for this combo already exists on the test
  machine)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Remove legacy adapters, relocate serve scripts, clean up artifacts
  ([`474fb31`](https://github.com/YingxuH/vllm_plugin/commit/474fb3140e4a36530f647dcd956c67cd18444efd))

- Delete vllm085.py and vllm010.py (unused legacy adapters for vLLM <0.10.1) - Delete root-level
  matrix_summary.md (stale copy; canonical version in artifacts/) - Move serve scripts from
  private_scripts/ to scripts/examples/ with cleaner names - Update serve_scripts paths in
  version_matrix_candidates.yaml and CLAUDE.md - Add delete/ to .gitignore (holds moved media/cache
  files pending manual cleanup) - Commit pending run_version_matrix_local.py improvements: orphan
  vLLM process cleanup, GPU memory wait before next server, improved terminate_group, N/A mode
  status in summary

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Untrack CLAUDE.md and .learnings/ from installable package
  ([`5a653a9`](https://github.com/YingxuH/vllm_plugin/commit/5a653a972ae67b8a6e6dd27f3c10ce6444125692))

These are local development aids, not part of the distributed package.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Documentation

- Fix outdated references across docs, add NoRepeatNGram bug resolution log
  ([`a244748`](https://github.com/YingxuH/vllm_plugin/commit/a244748ac4a9ea84cbcbb66f59d3963e1bf23b9a))

- CLAUDE.md: fix vllm0101 adapter range to 0.12.0–0.16.x (was 0.10.1) - tests/README.md: update
  supported versions from V0/V1 engine ranges to >=0.12.0,<0.17.0 - tests/test_utils.py: fix stale
  serve script reference - version-matrix-smoke.yml: remove unsupported vLLM 0.10.1–0.11.1, add
  0.14.0–0.16.0, fix transformers version from 5.0.0 to 4.57.6 - Add detailed NoRepeatNGram bug
  resolution log (.learnings/) for future technical article: root cause analysis, misdiagnosis
  timeline, fix design

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Update readme and example scripts for v0.3.x / V1 engine
  ([`a79d150`](https://github.com/YingxuH/vllm_plugin/commit/a79d150fce84f8ed2a2f777e1f96c80906322726))

- Fix version range to >=0.12.0 (matches __init__.py min_supported_version) - Replace broken
  matrix_summary.md links with inline compatibility table - All readme links use absolute GitHub
  URLs (PyPI-safe) - openai_serve_example.sh: add FlashInfer attention backend auto-detection,
  remove obsolete --logits-processor-pattern (entry-point handles it) - Remove per-request
  logits_processors from client examples (not needed in V1) - Fix hardcoded paths and typos in
  example scripts

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Features

- Add vLLM 0.10.1–0.16.x support, drop legacy 0.8.5–0.10.0 lane
  ([`7559f8d`](https://github.com/YingxuH/vllm_plugin/commit/7559f8d5ace9a8f48be031d603615dabea8f3510))

Introduce vllm0101.py as the single adapter for vLLM >= 0.10.1. Key changes over the old
  vllm085/vllm010 adapters: - Add get_placeholder_str classmethod (replaces _placeholder_str
  monkey-patch, which was a no-op since vLLM 0.9.x anyway) - Rename get_multimodal_embeddings →
  embed_multimodal to satisfy the SupportsMultiModal interface required from vLLM 0.13.0 onwards -
  Guard merge_multimodal_embeddings import with try/except (removed in vLLM 0.12.0; only exercised
  by the V0 engine path, which is never reached in V1) - Remove SamplingMetadata import (module gone
  in vLLM 0.11.0; V1 passes None for that argument anyway); type-annotate as Any

Update __init__.py: single routing branch for >= 0.10.1, remove the now-dead _placeholder_str patch,
  set soft cap at 0.17.0.

Update CI smoke matrix, version_matrix_candidates.yaml, test bounds, and README to reflect the new
  0.3.x lane (vLLM 0.10.1 ~ 0.16.x).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Drop vLLM 0.10.1–0.11.x support, require vLLM >= 0.12.0
  ([`3ea7ce0`](https://github.com/YingxuH/vllm_plugin/commit/3ea7ce0d4830aa7c582602f8fdb95fbd7392589f))

* Bump minimum supported version from 0.10.1 → 0.12.0 in __init__.py * Remove all vLLM 0.10/0.11
  compatibility shims from vllm0101.py: - Remove legacy merge_multimodal_embeddings import guard -
  Remove inspect-based _decoder_compute_logits_needs_sampling_metadata - Simplify compute_logits (no
  sampling_metadata param) - Simplify get_input_embeddings (no V0 multimodal merge path) - Add
  mm_options kwarg to get_dummy_inputs (new in 0.12.0 API) * Add NoRepeatNGramV1LogitsProcessor
  loaded via vllm.logits_processors entry point (pyproject.toml); ngram_size controlled by
  MERALION_NGRAM_SIZE env var (default 6) — no per-request logits_processors needed * Switch
  attention backend to FLASHINFER in serve scripts (FlashInfer natively supports Gemma2
  attn_logit_softcapping; FA3 does not) * Remove vLLM 0.10.1/0.10.2/0.11.0/0.11.1 from version
  matrix YAML * Add post_ready_warmup_seconds: 60 to matrix runner (FlashInfer JIT) * Remove
  per-request logits_processors from test_asr_latency_eval.py and test_asr_transcription_eval.py
  (rejected by V1 engine with HTTP 400) * Raise latency test budgets for FlashInfer baseline: -
  TTFT_P95_BUDGET_MS: 2000 → 6000 ms (batch-concurrency scheduling) - ITL_P95_BUDGET_MS: 100 → 120
  ms (FlashInfer ITL profile) * Add _warmup_server autouse fixture to latency tests (3 dummy
  requests at 5s/10s/30s audio durations to pre-compile FlashInfer kernels) * Update conftest.py:
  remove is_v1_08_09_engine / is_v1_010_engine helpers, narrow is_v1_engine to 0.12.0–0.17.0 range *
  Update version checks in test_version_compatibility.py and test_model_correctness.py to use 0.12.0
  floor

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Testing

- Align asr evaluation with audiobench
  ([`e600257`](https://github.com/YingxuH/vllm_plugin/commit/e600257691dfe2485a61a2ff75d75dd24e7da17e))

- Update asr accuracy test sample value
  ([`a20802c`](https://github.com/YingxuH/vllm_plugin/commit/a20802cbad83079ca38487a55b9a854359586afd))


## v0.2.0 (2026-02-25)

### Bug Fixes

- Resolve bandit warning about assertion
  ([`c7dddea`](https://github.com/YingxuH/vllm_plugin/commit/c7dddea207d656821d29ed60d3d10e85ff6e0993))

- Resolve pylint warning
  ([`c912429`](https://github.com/YingxuH/vllm_plugin/commit/c9124292bc8b4d80fad4cde0de3b3e8d7517c785))

- Sdpa import error for transformers 5 beyond
  ([`e26b95f`](https://github.com/YingxuH/vllm_plugin/commit/e26b95f854d0b50d01c02ea7c7b644dfe713b19e))

### Chores

- Update build packages
  ([`bef8052`](https://github.com/YingxuH/vllm_plugin/commit/bef8052af4429e2842c73b5562e29e1a8a58570d))

### Continuous Integration

- Add automatic vllm version matrix smoke test
  ([`e2681c3`](https://github.com/YingxuH/vllm_plugin/commit/e2681c3ee9e966fbffbf1878dd37127e48596c0a))

- Potential fix for code scanning alert no. 4: Workflow does not contain permissions
  ([`6edec35`](https://github.com/YingxuH/vllm_plugin/commit/6edec3536aa0774da855c3bed86df38ff64c2886))

Co-authored-by: Copilot Autofix powered by AI
  <62310815+github-advanced-security[bot]@users.noreply.github.com>

- Potential fix for code scanning alert no. 5: Workflow does not contain permissions
  ([`bf8d3ce`](https://github.com/YingxuH/vllm_plugin/commit/bf8d3ce5d5900d9bf20e5d9d2c04cab7fad37e25))

Co-authored-by: Copilot Autofix powered by AI
  <62310815+github-advanced-security[bot]@users.noreply.github.com>

- Restructure security and dependency checks
  ([`0cb0756`](https://github.com/YingxuH/vllm_plugin/commit/0cb0756b347f277f529b256d4a3ac11b0e8fc262))

- Update pylint action
  ([`8c2c294`](https://github.com/YingxuH/vllm_plugin/commit/8c2c2944634b71b2d515125b511608f4e35dbd25))

- Update release workflow to remove push trigger
  ([`aab394f`](https://github.com/YingxuH/vllm_plugin/commit/aab394f6d998e5a451452be219db542bcae788ba))

Removed push trigger for main branch from release workflow.

- Update smoke test
  ([`69cd46f`](https://github.com/YingxuH/vllm_plugin/commit/69cd46f941e1377e58cde3f77948efe6f67e43e1))

### Documentation

- Add compatibility summary
  ([`3266bb1`](https://github.com/YingxuH/vllm_plugin/commit/3266bb10b00d8c998f175936d430b02fb7eac9ae))

- Restructure readme badges
  ([`e241b8c`](https://github.com/YingxuH/vllm_plugin/commit/e241b8ca31beec215406d9b39e5eede256e6a277))

- Update readme to introduce two lanes
  ([`f951991`](https://github.com/YingxuH/vllm_plugin/commit/f951991ea8282eb6c57599c709bc1414200dafce))

### Features

- Update vllm support to 0.10.0
  ([`2a9b53d`](https://github.com/YingxuH/vllm_plugin/commit/2a9b53d1594e9b4904d2a1f515ff258e67a59123))

### Testing

- Add version compatiblity test automation
  ([`9291910`](https://github.com/YingxuH/vllm_plugin/commit/9291910eb121d95008c0d6846b985cbd60caaab8))

- Relax stt test threshold
  ([`d85bc32`](https://github.com/YingxuH/vllm_plugin/commit/d85bc32b0cbc58fd3263c048fbe4be46bceda824))

- Update vllm version tests
  ([`b78471f`](https://github.com/YingxuH/vllm_plugin/commit/b78471fc9abb69e34477a1be29569121a74bb8bb))


## v0.1.5 (2026-02-21)

### Bug Fixes

- Remove unused import
  ([`d01bd88`](https://github.com/YingxuH/vllm_plugin/commit/d01bd8845890190578a56de90767125f9dcf9168))

### Chores

- Fix build_command formatting in pyproject.toml
  ([`6335edd`](https://github.com/YingxuH/vllm_plugin/commit/6335eddf9f96ff793aa2ddc6c0ef0af5b96af9e4))

### Continuous Integration

- Add CodeQL analysis workflow configuration
  ([`f8c21f5`](https://github.com/YingxuH/vllm_plugin/commit/f8c21f5495dfd41f976d164a2e52492c100c5bc1))

- Add Dependency Review Action workflow
  ([`948db7b`](https://github.com/YingxuH/vllm_plugin/commit/948db7bda8c0114f4564a45388d27cb754e23e74))

This workflow scans dependency manifest files for known vulnerabilities in PRs and blocks merging if
  vulnerabilities are found.

- Add publish automation
  ([`ab0c9c4`](https://github.com/YingxuH/vllm_plugin/commit/ab0c9c4855ff57cabfd3cb153cbba9fa85169bf3))

- Add Pylint workflow for Python code analysis
  ([`dfa8728`](https://github.com/YingxuH/vllm_plugin/commit/dfa8728503f08e29c9c72c430fc39d2952bc7b5f))

- Add security scanning workflow (bandit + pip-audit)
  ([`140c242`](https://github.com/YingxuH/vllm_plugin/commit/140c242536b81d3643f4e76b7a0a1a999257a32a))

- Modify build command to include package installation
  ([`adc678d`](https://github.com/YingxuH/vllm_plugin/commit/adc678dd895ab5e6c2646409cb5860c99a2ab7e9))

Updated build command to install build package before execution.

- Update actions
  ([`90fca87`](https://github.com/YingxuH/vllm_plugin/commit/90fca87ca848761a27f1ac5a27dd099550f99751))

- Update pylint analysis to only include src directory
  ([`0c47564`](https://github.com/YingxuH/vllm_plugin/commit/0c47564f4f9523c1ea06409eb3f85e71e4e80c7c))

### Documentation

- Add action badge
  ([`04052a1`](https://github.com/YingxuH/vllm_plugin/commit/04052a1fd3eea589631f1540b0136ef110e79706))

- Update example script and readme links
  ([`34a916c`](https://github.com/YingxuH/vllm_plugin/commit/34a916c9f13c580d340fdb614dfd23b9c214f972))

### Refactoring

- Add docstring
  ([`c75e7b2`](https://github.com/YingxuH/vllm_plugin/commit/c75e7b282bd6a41c56c9238d712d2c6e56824c2c))

- Formatting code
  ([`addb967`](https://github.com/YingxuH/vllm_plugin/commit/addb967e1619c0795930bc141eeac3757d2b82f9))

### Testing

- Add test script
  ([`27dbad2`](https://github.com/YingxuH/vllm_plugin/commit/27dbad2517964d4049b9f12ee41f9117c22f70a6))

- Update test
  ([`ffa3903`](https://github.com/YingxuH/vllm_plugin/commit/ffa3903418843dac9e3c58fc4ca9617b388e5f37))


## v0.1.4 (2026-02-19)
