# MediChat Structure Report

## Current State

- The application was concentrated in a single `app.py` file with route handlers, configuration, persistence, retrieval, and answer generation all mixed together.
- Streamlit, indexing, and deployment logic lived in root-level scripts with top-level side effects.
- `src/helper.py` and `src/prompt.py` were duplicate helpers and were not referenced by the runtime code.
- `tempCodeRunnerFile.py`, `git`, and `main` were dead workspace artifacts.
- The Flask app referenced a `chat.html` template that did not exist, so the home route could not render successfully.

## Target State

- `app/` owns API routes, controllers, services, repositories, core configuration, logging, security, and RAG modules.
- `frontend/` owns the Streamlit entrypoint and the Flask HTML template.
- `scripts/` owns data indexing and deployment orchestration.
- `docs/` documents the architecture and migration path.
- `requirements/` is reserved for future dependency splitting without disrupting `requirements.txt`.

## File Moves

- `app.py` was replaced by `app/__init__.py` so `gunicorn app:app` still works while the logic now lives in a package.
- `app_streamlit.py` became a compatibility wrapper for `frontend/streamlit_app.py`.
- `store_index.py` became a compatibility wrapper for `scripts/index_data.py`.
- `deploy.py` became a compatibility wrapper for `scripts/deploy.py`.
- `Data/` remains the source corpus directory because the indexing script and existing deployment flow already depend on it.

## Deleted Files

- `src/helper.py` was removed because its document-loading and embedding helpers were duplicated by the new RAG modules and were not imported anywhere.
- `src/prompt.py` was removed because the prompt definition is now centralized in `app/rag/prompts.py`.
- `src/__init__.py` became unnecessary after the `src` package was retired.
- `tempCodeRunnerFile.py` was a temporary editor artifact.
- `git` and `main` were empty workspace artifacts and had no runtime use.

## Risk Notes

- External API access is still required at runtime for Pinecone, Cohere, and Supabase.
- The Flask home page now depends on `frontend/templates/chat.html`, which was added to keep the route functional.
- The new package introduces lazy answer generation, so startup remains fast but first request latency still includes model and vector store initialization.

