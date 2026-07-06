# MediChat Frontend

Angular 21 frontend for the MediChat medical learning assistant.

## Run

```bash
cd frontend
npm.cmd install
npm.cmd start
```

## Build

```bash
cd frontend
npm.cmd run build
```

## Connected areas

- Landing page
- Login, signup, and logout
- Live chat against the Flask backend
- Theme toggle with light, dark, and system modes
- PWA shell and offline assets

## Honest shells

These pages are intentionally minimal until real backend contracts exist:

- Dashboard
- Study tools
- Flashcards
- Quizzes
- Profile
- Settings
- Analytics

## Backend integration

When the app runs on `localhost:4200`, requests go to `http://localhost:8000`.
When Flask serves the built frontend, the app stays on the same origin.

The backend must allow CSRF and credentialed requests from the Angular dev host.

## Notes

- Chat uses the Flask `/get` endpoint with `msg` form data.
- Auth uses JSON to `/login` and `/signup`.
- `GET /health` boots the CSRF cookie and checks backend readiness.
