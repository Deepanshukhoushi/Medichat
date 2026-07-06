# Dependency Report

## Runtime Dependencies

- `Flask` powers the web app.
- `streamlit` powers the alternate chat UI.
- `langchain`, `langchain-core`, `langchain-community`, `langchain-pinecone`, and `langchain-cohere` power retrieval and generation.
- `supabase` handles authentication and persistence.
- `pydantic` validates auth payloads.
- `python-dotenv` loads environment variables.
- `pypdf` loads the indexed PDF corpus.

## Dependency Hygiene

- The dependency list now uses upper bounds in `requirements.txt`.
- `requirements/lock.txt` captures a reproducible pinned snapshot.
- `pydantic` is explicitly declared because the code imports it directly.

