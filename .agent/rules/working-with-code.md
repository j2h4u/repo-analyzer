---
trigger: always_on
---

# Python guidelines

## venv

The project uses python3 and venv.
A venv activation status can be checked with a simple bash command `[[ -n "$VIRTUAL_ENV" ]] && echo 'venv activated' || echo 'venv is not active'`.

## Code style

Always document via docstrings.
Use pylint. It's already installed into the venv.

## Other

Add shebang `#!` and then `chmod +x` them to all scripts that can be executed on its own, so you don't need to run them with `python3 -c foobar.py`.
