# Contributing to Live Translator

Thank you for considering contributing to Live Translator! This document explains how to set up your development environment, the conventions we follow, and how to submit changes.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
  - [Branch Naming](#branch-naming)
  - [Making Changes](#making-changes)
  - [Running Tests](#running-tests)
  - [Linting and Formatting](#linting-and-formatting)
- [Code Style](#code-style)
  - [Python](#python)
  - [FastAPI / WebSocket Patterns](#fastapi--websocket-patterns)
  - [Async Code](#async-code)
  - [Templates / Frontend](#templates--frontend)
- [Commit Messages](#commit-messages)
- [Pull Requests](#pull-requests)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Security Vulnerabilities](#security-vulnerabilities)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Report unacceptable behavior to the maintainers.

## Getting Started

### Prerequisites

- **Python 3.10+** (the codebase uses `X | None` union syntax)
- **Git**
- A microphone-capable browser for manual testing (Chrome or Edge recommended)

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/<your-username>/sr_live-translator.git
cd sr_live-translator

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install runtime + dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment
cp .env.example .env
# Edit .env — set at least APP_PASSWORD
```

Verify everything works:

```bash
# Run the test suite
pytest

# Quick syntax check
python -m compileall app tests

# Start the dev server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Development Workflow

### Branch Naming

Use descriptive branch names with a category prefix:

```
feat/add-deepl-translation
fix/websocket-reconnect-loop
docs/update-api-reference
refactor/extract-audio-worklet
test/elevenlabs-happy-path
```

### Making Changes

1. Create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature main
   ```
2. Make your changes in small, focused commits.
3. Run the test suite and fix any failures before pushing.
4. Push your branch and open a pull request.

### Running Tests

```bash
# Run all tests (quick, quiet output — configured in pytest.ini)
pytest

# Verbose output
pytest -vv

# Run a single test file
pytest tests/test_main.py

# Run a specific test by name
pytest tests/test_main.py::test_ws_translates_text

# Run tests matching a substring
pytest -k translate

# Run with coverage
pytest --cov=app --cov-report=term-missing
```

**Known test issues:** Three Deepgram-related tests (`test_ws_deepgram_happy_path_emits_interim_and_final`, `test_ws_deepgram_init_failure_sends_error`, `test_ws_deepgram_missing_sdk_returns_error`) may hang due to threading/SDK interactions. If you are working on Deepgram code, run these tests individually and be prepared to terminate them manually if they hang.

### Linting and Formatting

No linter is pinned in the project yet. If you have them installed:

```bash
# Ruff (recommended)
ruff check .
ruff format .

# mypy (optional — expect some noise from googletrans/deepgram typing)
mypy app
```

Keep formatting changes minimal and local. Avoid sending PRs that reformat entire files without functional changes.

## Code Style

### Python

- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants. Use a leading underscore for internal helpers (`_translate`, `_sign`).
- **Imports:** Group with blank lines between: stdlib, third-party, local.
- **Strings:** Prefer f-strings over `str.format()`.
- **Type hints:** Use `X | None` union syntax (Python 3.10+). Add explicit return types for non-trivial helpers. Use `TypedDict` for JSON payload shapes.
- **Error handling:** Prefer specific exceptions. Use broad `except Exception` only at outer boundaries (WebSocket loops, optional imports, translation calls).
- **Logging:** Log actionable context. Never log secrets (passwords, API keys, auth tokens).

### FastAPI / WebSocket Patterns

- Keep route handlers small; push logic into helper functions.
- WebSocket endpoints must:
  1. Validate auth and origin **before** calling `accept()`.
  2. Send structured JSON errors (`{"error": "..."}`) when possible.
  3. Close with appropriate codes (`1008` for policy/unauthorized, `1011` for server error).
- Treat `send`/`close` failures as best-effort (wrap in try/except).

### Async Code

- **Never block the event loop.** If a library call is sync-only, offload with `asyncio.to_thread(...)`.
- Apply timeouts with `asyncio.wait_for(...)`.
- When receiving events from other threads, use `loop.call_soon_threadsafe(...)`.
- Catch `asyncio.CancelledError` during shutdown paths.

### Templates / Frontend

- Templates are Jinja HTML (`app/templates/*.html`) with inline CSS and inline JS.
- Keep accessibility attributes (ARIA labels, roles, skip links).
- Avoid large formatting-only diffs in HTML/CSS/JS.
- Use `textContent` / `createTextNode` instead of `innerHTML` to prevent XSS.
- Wrap all `JSON.parse` calls in try/catch in WebSocket `onmessage` handlers.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

<optional body>
```

**Types:**

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `chore` | Build process, CI, dependencies |
| `style` | Formatting, whitespace (no logic change) |

**Examples:**

```
feat(stt): add ElevenLabs Scribe v2 Realtime engine
fix(ws): prevent stale interim translations from blocking finals
docs: add CONTRIBUTING.md and issue templates
perf: throttle interim translations to prevent queue buildup
```

- Keep the first line under 72 characters.
- Use the imperative mood ("add", not "added" or "adds").
- Reference issues when applicable: `Fixes #42`.

## Pull Requests

1. **One concern per PR.** Don't mix features, bug fixes, and refactors in the same PR.
2. **Write a clear description.** Explain *what* changed and *why*.
3. **Include tests** for new functionality or bug fixes when feasible.
4. **Ensure tests pass** before requesting review.
5. **Keep diffs minimal.** Avoid unrelated formatting changes.

### PR Checklist

Before submitting, verify:

- [ ] Tests pass (`pytest`)
- [ ] No syntax errors (`python -m compileall app tests`)
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) format
- [ ] New environment variables are documented in `.env.example`
- [ ] No secrets or credentials in the diff
- [ ] Breaking changes are clearly noted in the PR description

## Reporting Bugs

Open a [Bug Report](../../issues/new?template=bug_report.yml) and include:

- Steps to reproduce
- Expected vs. actual behavior
- Browser, OS, and Python version
- STT engine in use
- Relevant log output (redact any API keys)

## Requesting Features

Open a [Feature Request](../../issues/new?template=feature_request.yml) and describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Security Vulnerabilities

**Do not open a public issue for security vulnerabilities.** See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.
