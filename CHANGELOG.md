# CHANGELOG


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
