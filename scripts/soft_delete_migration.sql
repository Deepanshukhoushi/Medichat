-- ============================================================
-- MediChat — Soft Delete Migration
-- Run this script in your Supabase SQL editor.
-- ============================================================

ALTER TABLE conversations ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
