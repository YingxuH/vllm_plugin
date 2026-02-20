# Changelog

All notable changes to this project are documented in this file.

## [0.1.4] - 2026-02-18

### Fixed
- Fixed a bug where a single request containing more than one audio input could fail.
- Fixed a bug where batching requests with different audio chunk counts could cause an internal vLLM server failure.

### Changed
- Added and improved docstrings across key code paths for maintainability and clarity.

## [0.1.3] - Previous release
- Initial public release for MERaLiON2 vLLM plugin architecture registration and serving support.
