---
trigger: model_decision
description: When working with Python code and *.py files
---

# Python guidelines

## venv
The project uses python3 and venv. A venv activation status can be checked with a simple bash command:
```bash
[[ -n "$VIRTUAL_ENV" ]] && echo 'venv activated' || echo 'venv is not active'
```

## Documentation
Always document scripts, modules, functions, classes, etc. with docstrings.

## Linting
Use `pylint`. It's already installed into the current venv.

## Other
Add shebang `#!` and then `chmod +x` them to all scripts that can be executed on its own, so you don't need to run them with `python3 -c foobar.py`.