# Deleted Files Report

## Removed From Runtime

- `app.py` was replaced by the package entrypoint in `app/__init__.py`.
- `src/helper.py` was removed after proving it had no live call sites and its logic was consolidated into `app/rag/`.
- `src/prompt.py` was removed after prompt duplication was centralized in `app/rag/prompts.py`.
- `src/__init__.py` became unnecessary once `src/` stopped being a runtime package.
- `tempCodeRunnerFile.py` was an editor artifact.

## Untracked From Git

- `.env` was removed from git tracking, but the local file was left in place.
- `__pycache__/` bytecode files were removed from git tracking.
- `venv/` was removed from git tracking while keeping the local environment.

