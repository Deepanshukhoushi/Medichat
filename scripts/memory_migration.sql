-- ============================================================
-- MediChat — Advanced Conversation Memory Migration
-- Run this script in your Supabase SQL editor.
-- Existing tables (chat_history, conversations) are NOT touched.
-- ============================================================

-- 1. chat_messages: full per-session message log with content column
CREATE TABLE IF NOT EXISTS chat_messages (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     text        NOT NULL,
    session_id  text        NOT NULL,
    role        text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content     text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user
    ON chat_messages (user_id);

-- 2. session_topics: current learning topic per session
CREATE TABLE IF NOT EXISTS session_topics (
    session_id      text        PRIMARY KEY,
    current_topic   text,
    related_topics  jsonb       NOT NULL DEFAULT '[]'::jsonb,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- 3. conversation_summaries: rolling study summaries (generated every ~10 messages)
CREATE TABLE IF NOT EXISTS conversation_summaries (
    session_id  text        PRIMARY KEY,
    summary     text        NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- ============================================================
-- Row-Level Security (optional but recommended)
-- Enable RLS and add permissive policies if you use Supabase Auth.
-- ============================================================
-- ALTER TABLE chat_messages          ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE session_topics         ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE conversation_summaries ENABLE ROW LEVEL SECURITY;
