# DevContainer Changelog

## v0.1.1 - 2024-12-29

### Changed
- **Fixed Codespaces build failure**: Changed base image from `mcr.microsoft.com/devcontainers/python:3.11` to `mcr.microsoft.com/devcontainers/python:0-3.11-bookworm`

### Root Cause
The original devcontainer used `python:3.11` which defaults to Debian **trixie** (testing). The `docker-in-docker` feature attempted to install **moby** packages, which are not available for trixie. The Codespaces build failed with:
```
E: Package 'moby-cli' has no installation candidate
```

### Solution
Explicitly use `bookworm` (Debian 12 stable) which has full moby package support:
- Image: `mcr.microsoft.com/devcontainers/python:0-3.11-bookworm`
- Feature: `docker-in-docker:2` with `moby: true`

### Additional Changes
- Added `workspaceFolder: /workspaces/MarketResearchAI`
- Added GitLens extension
- Changed `postCreateCommand` to install from `requirements-dev.txt` (lightweight)
- Added `TRANSFORMERS_OFFLINE=1` to containerEnv

## v0.1.0 - 2024-12-29

### Initial Release
- Base Python 3.11 devcontainer
- Docker-in-docker feature
- VS Code extensions for Python development
