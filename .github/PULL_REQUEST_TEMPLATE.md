## What changed?

<!-- Describe the change and why it is needed. -->

## How to test

- [ ] `pytest`
- [ ] `python -m compileall app tests`
- [ ] Manual smoke test in the browser (if relevant)

## Checklist

- [ ] No secrets in the diff (`.env`, API keys, auth cookies)
- [ ] New env vars are documented in `.env.example` and `README.md`
- [ ] WebSocket endpoints validate auth/origin before `accept()` (if touched)
- [ ] Templates use `textContent`/`createTextNode` (no `innerHTML`) (if touched)
