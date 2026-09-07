# Security Architecture

This document records deliberate architectural security decisions so that contributors,
operators, and security reviewers understand the threat model and isolation boundaries.

---

## Supabase Security Model

### Current Architecture — Service-Role Client

The application creates a **single, shared Supabase client** at startup using the
`service_role` secret key (see `ServiceFactory.__init__`).  The same client is reused
for every request and every user.

**Consequence: Row Level Security (RLS) is bypassed.**

Supabase documents this explicitly: the service-role key bypasses all RLS policies.
This means every `CREATE POLICY ... USING (auth.uid()::text = user_id)` statement in
the migration scripts (`scripts/*.sql`) is **inert at runtime** — those policies run
in a context where `auth.uid()` always resolves to `NULL` for a service-role client.

### What This Means in Practice

The **sole** tenant-isolation boundary is the application-level `.eq("user_id", user_id)`
filter present in each repository query.

| What protects data | Status |
|---|---|
| Supabase RLS policies | Inert — service-role key bypasses them |
| `.eq("user_id", user_id)` in repository queries | Active — this is the real barrier |

### Known Risk: Repository Methods Without `user_id` Filters

Any repository method that omits the `user_id` filter is a direct, unmitigated IDOR
with no RLS fallback.  All known instances were identified and fixed in the Round 1 audit:

| Method | Fix applied |
|---|---|
| `FlashcardRepository.rate_card` | Ownership verified in-query (Issue 6) |
| `ChatHistoryRepository.delete_latest_exchange` | Added user_id filter (Issue 7) |
| `ConversationRepository.update_title` | Added user_id filter (Issue 8) |

Any future repository method that mutates or reads per-user data **must** include
`.eq("user_id", user_id)` in the same query.

### Why `auth.admin.*` Requires the Service-Role Key

`ProfileRepository.get_profile()` calls `self.supabase.auth.admin.get_user_by_id()`.
The `auth.admin` namespace is only available with the service-role key.  This is the
architectural reason the application cannot simply switch to the anon key.

### Path to True RLS Enforcement

1. Store the user JWT in `g.session_context.access_token`.
2. Before each repository call, create a request-scoped client via
   `supabase.auth.set_session(access_token, refresh_token)`.
3. Move `auth.admin.*` calls to a separate admin-scoped client never used for data queries.

This is a significant refactor. The application-level `user_id` filter approach provides
equivalent protection with lower complexity for the current scale.

---

## Session Cookie and CSRF

- Session cookies: `HttpOnly`, `Secure` (production), `SameSite=None` (cross-origin) / `Lax` (dev).
- CSRF enforced via double-submit cookie for same-origin POST/PUT/PATCH requests.
- Cross-origin requests from whitelisted `FRONTEND_ORIGINS` skip CSRF (protected by CORS preflight).

---

## Rate Limiting

All auth, content-generation, and mutating write endpoints are rate-limited via Redis.
See `app/core/security/web.py` for current limits.

---

## Audit Logging

Security events are asynchronously written to `audit_log` via Celery with a synchronous
daemon-thread fallback when the broker is unavailable. Failures logged at ERROR level.
