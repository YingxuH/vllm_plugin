
### Local Compatibility Matrix Automation

Use the matrix runner to automatically test `vLLM`/`transformers` combinations across all 3 serve modes (`general`, `ctm`, `asr`) and run mode-specific `pytest` suites.

Run the full configured matrix:

```bash
python3 scripts/compatibility/run_version_matrix_local.py
```

Run only selected versions:

```bash
python3 scripts/compatibility/run_version_matrix_local.py \
  --only-vllm 0.9.2 \
  --only-transformers 4.51.3 4.55.4
```

Resume a previously interrupted run:

```bash
python3 scripts/compatibility/run_version_matrix_local.py --resume
```

Edit candidate versions and test mapping in:

- `scripts/compatibility/version_matrix_candidates.yaml`

The runner generates:

- `artifacts/matrix_results.json` (machine-readable per-combination details)
- `artifacts/matrix_summary.md` (quick compatibility summary)
- `artifacts/logs/<vllm>/<transformers>/<mode>.log` (serve/pytest logs)

If your latest run only covered a subset (for example only `transformers==5.0.0`), rebuild a full compatibility view from all historical artifacts logs:

```bash
python3 scripts/compatibility/generate_summary_from_artifacts.py
```

Compatibility report file:

- `artifacts/compatibility_summary.md`

Notes:

- The runner creates isolated virtualenvs under `.venvs/matrix`.
- It requires port `8063` to be free before each serve mode starts.
- Compatibility is reported per tuple `(vllm_version, transformers_version, mode)`.