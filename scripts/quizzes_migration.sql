-- ============================================================
-- MediChat — Quizzes Migration
-- Run this script in your Supabase SQL editor.
-- ============================================================

CREATE TABLE IF NOT EXISTS quiz_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    topic text NOT NULL,
    questions jsonb NOT NULL,
    score int,
    completed_at timestamptz,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_quiz_sessions_user
    ON quiz_sessions (user_id);
