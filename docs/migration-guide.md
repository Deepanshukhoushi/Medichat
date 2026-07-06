# Migration Guide

## For Developers

1. Copy `.env.example` to `.env`.
2. Fill in `PINECONE_API_KEY`, `COHERE_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY`.
3. Optionally set `FLASK_SECRET_KEY`, `SESSION_HISTORY_TTL_SECONDS`, and `REDIS_URL`.
4. Install dependencies from `requirements.txt`.
5. Run `python store_index.py` if the PDF corpus changes.
6. Start the app with `gunicorn app:app --bind 0.0.0.0:8000` or `python app_streamlit.py`.

## For Operations

- Keep local datasets outside the repo for future large-corpus deployments.
- Prefer Redis for session history in multi-worker deployments.
- Use the files under `docs/` as the source of truth for architecture, security, and data handling decisions.

