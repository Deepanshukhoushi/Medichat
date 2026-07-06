-- ============================================================
-- MediChat — Profiles Migration
-- Run this script in your Supabase SQL editor.
-- ============================================================

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id text PRIMARY KEY,
    display_name text,
    medical_year int,   -- 1–6
    specialty text,
    university text,
    updated_at timestamptz DEFAULT now()
);
