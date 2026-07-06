# MediChat

MediChat is a medical learning assistant with a Flask backend and an Angular 21 frontend.

## What is connected now

- `GET /health` returns backend status and sets the CSRF cookie.
- `POST /get` sends real chat prompts to the Flask RAG backend.
- `POST /login`, `POST /signup`, and `POST /logout` are wired from the Angular auth screens.
- The Flask app can serve the built Angular frontend from `frontend/dist/medichat-frontend/browser`.

## Run the backend

```bash
pip install -r requirements.txt
python -m app
```

## Run the frontend in development

```bash
cd frontend
npm.cmd install
npm.cmd start
```

The frontend dev server targets `http://localhost:8000` and the Flask backend allows credentials + CSRF headers from `http://localhost:4200`.

## Build the frontend

```bash
cd frontend
npm.cmd run build
```

## Notes

- Required backend environment variables still include `PINECONE_API_KEY`, `COHERE_API_KEY`, `FLASK_SECRET_KEY`, and Supabase settings when persistence is enabled.
- Set `FRONTEND_ORIGINS` if you want to allow additional Angular dev hosts.
- The remaining study-tool pages are honest shells until their backend endpoints exist.

## Frontend docs

- [Frontend README](frontend/README.md)
- [Architecture](docs/architecture.md)
- [Migration guide](docs/migration-guide.md)
