# Security Improvements

## Changes

- `.env` is no longer tracked by git.
- `.gitignore` now excludes local environment files, caches, virtual environments, IDE settings, data, and logs.
- Auth payloads are validated before reaching Supabase.
- Session history can be stored in Redis when `REDIS_URL` is configured.
- Secret-rotation guidance is documented in `docs/secret-rotation-checklist.md`.

## Notes

- Historical git commits may still contain previously committed secrets.
- Those secrets should be rotated through the provider consoles.

