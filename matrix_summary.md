# vLLM + Transformers Compatibility (From Artifacts)

Generated: `2026-02-24T20:02:22.247940+00:00`

Legend: `PASS` = test mode succeeded, `FAIL` = mode failed, `NA` = mode not run (maybe due to installation failure), `INCOMPLETE` = partial run.

## vLLM 0.8.5

| transformers | install | general | latency | asr | overall | signature |
|---|---|---|---|---|---|---|
| 4.51.1 | PASS | PASS | PASS | PASS | PASS |  |
| 4.51.3 | PASS | PASS | PASS | PASS | PASS |  |
| 4.52.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.53.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.54.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.55.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.55.4 | PASS | PASS | PASS | PASS | PASS |  |
| 4.56.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.57.6 | PASS | PASS | PASS | PASS | PASS |  |
| 5.0.0 | PASS | FAIL | NA | NA | FAIL | AttributeError: TokenizersBackend has no attribute all_special_tokens_extended. Did you mean: 'num_special_tokens_to_add'? |

- Fully passing transformers: 4.51.1, 4.51.3, 4.52.0, 4.53.0, 4.54.0, 4.55.0, 4.55.4, 4.56.0, 4.57.6

## vLLM 0.9.0

| transformers | install | general | latency | asr | overall | signature |
|---|---|---|---|---|---|---|
| 4.51.1 | PASS | PASS | PASS | PASS | PASS |  |
| 4.51.3 | PASS | PASS | PASS | PASS | PASS |  |
| 4.52.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.53.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.54.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.55.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.55.4 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.56.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.57.6 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 5.0.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |

- Fully passing transformers: 4.51.1, 4.51.3, 4.52.0, 4.53.0

## vLLM 0.9.1

| transformers | install | general | latency | asr | overall | signature |
|---|---|---|---|---|---|---|
| 4.51.1 | PASS | PASS | PASS | PASS | PASS |  |
| 4.51.3 | PASS | PASS | PASS | PASS | PASS |  |
| 4.52.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.53.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.54.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.55.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.55.4 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.56.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.57.6 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 5.0.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |

- Fully passing transformers: 4.51.1, 4.51.3, 4.52.0, 4.53.0

## vLLM 0.9.2

| transformers | install | general | latency | asr | overall | signature |
|---|---|---|---|---|---|---|
| 4.51.1 | PASS | PASS | PASS | PASS | PASS |  |
| 4.51.3 | PASS | PASS | PASS | PASS | PASS |  |
| 4.52.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.53.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.54.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.55.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.55.4 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.56.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 4.57.6 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |
| 5.0.0 | PASS | FAIL | NA | NA | FAIL | ValueError: 'aimv2' is already used by a Transformers config, pick another name. |

- Fully passing transformers: 4.51.1, 4.51.3, 4.52.0, 4.53.0

## vLLM 0.10.0

| transformers | install | general | latency | asr | overall | signature |
|---|---|---|---|---|---|---|
| 4.51.1 | FAIL | NA | NA | NA | FAIL | [2026-02-24T13:56:40.686425+00:00] EXIT: 1 |
| 4.51.3 | FAIL | NA | NA | NA | FAIL | [2026-02-24T13:56:44.408565+00:00] EXIT: 1 |
| 4.52.0 | FAIL | NA | NA | NA | FAIL | [2026-02-24T13:56:48.123028+00:00] EXIT: 1 |
| 4.53.0 | FAIL | NA | NA | NA | FAIL | [2026-02-24T13:56:52.224843+00:00] EXIT: 1 |
| 4.54.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.55.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.55.4 | PASS | PASS | PASS | PASS | PASS |  |
| 4.56.0 | PASS | PASS | PASS | PASS | PASS |  |
| 4.57.6 | PASS | PASS | PASS | PASS | PASS |  |
| 5.0.0 | PASS | FAIL | NA | NA | FAIL | AttributeError: TokenizersBackend has no attribute all_special_tokens_extended. Did you mean: 'num_special_tokens_to_add'? |

- Fully passing transformers: 4.54.0, 4.55.0, 4.55.4, 4.56.0, 4.57.6
