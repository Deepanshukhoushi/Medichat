-- ============================================================
-- MediChat — Audit Log Migration
-- Run this script in your Supabase SQL editor.
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type  text        NOT NULL,
    user_id     text,
    remote_addr text,
    details     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_created
    ON audit_log (user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_audit_log_event_created
    ON audit_log (event_type, created_at);

-- Optional: auto-purge rows older than 90 days via pg_cron (Supabase Pro)
-- SELECT cron.schedule('purge-audit-log', '0 3 * * *',
--   $$DELETE FROM audit_log WHERE created_at < now() - interval '90 days'$$);
