-- ============================================================
-- MediChat — Complete RLS Fix Migration
-- Run this ENTIRE script in your Supabase SQL editor.
-- It is idempotent (safe to run multiple times).
-- ============================================================

-- ----------------------------------------------------------------
-- FIX 1: Add missing deleted_at column to conversations table
--         (fixes 502 on GET /api/conversations)
-- ----------------------------------------------------------------
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS deleted_at timestamptz;


-- ================================================================
-- FIX 2: Row-Level Security policies for ALL tables
--
-- The backend uses the anon/authenticated key, so every table with
-- RLS enabled needs explicit policies for the authenticated role.
-- ================================================================

-- ----------------------------------------------------------------
-- conversations
-- ----------------------------------------------------------------
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='conversations' AND policyname='users can insert own conversations') THEN
    CREATE POLICY "users can insert own conversations" ON conversations FOR INSERT WITH CHECK (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='conversations' AND policyname='users can view own conversations') THEN
    CREATE POLICY "users can view own conversations" ON conversations FOR SELECT USING (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='conversations' AND policyname='users can update own conversations') THEN
    CREATE POLICY "users can update own conversations" ON conversations FOR UPDATE USING (auth.uid()::text = user_id) WITH CHECK (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='conversations' AND policyname='users can delete own conversations') THEN
    CREATE POLICY "users can delete own conversations" ON conversations FOR DELETE USING (auth.uid()::text = user_id);
  END IF;
END $$;


-- ----------------------------------------------------------------
-- user_profiles
-- ----------------------------------------------------------------
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='user_profiles' AND policyname='users can insert own profile') THEN
    CREATE POLICY "users can insert own profile" ON user_profiles FOR INSERT WITH CHECK (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='user_profiles' AND policyname='users can view own profile') THEN
    CREATE POLICY "users can view own profile" ON user_profiles FOR SELECT USING (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='user_profiles' AND policyname='users can update own profile') THEN
    CREATE POLICY "users can update own profile" ON user_profiles FOR UPDATE USING (auth.uid()::text = user_id) WITH CHECK (auth.uid()::text = user_id);
  END IF;
END $$;


-- ----------------------------------------------------------------
-- chat_messages (user_id col)
-- ----------------------------------------------------------------
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='chat_messages' AND policyname='users can insert own messages') THEN
    CREATE POLICY "users can insert own messages" ON chat_messages FOR INSERT WITH CHECK (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='chat_messages' AND policyname='users can view own messages') THEN
    CREATE POLICY "users can view own messages" ON chat_messages FOR SELECT USING (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='chat_messages' AND policyname='users can delete own messages') THEN
    CREATE POLICY "users can delete own messages" ON chat_messages FOR DELETE USING (auth.uid()::text = user_id);
  END IF;
END $$;


-- ----------------------------------------------------------------
-- flashcard_decks (user_id col)
-- ----------------------------------------------------------------
ALTER TABLE flashcard_decks ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='flashcard_decks' AND policyname='users can insert own decks') THEN
    CREATE POLICY "users can insert own decks" ON flashcard_decks FOR INSERT WITH CHECK (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='flashcard_decks' AND policyname='users can view own decks') THEN
    CREATE POLICY "users can view own decks" ON flashcard_decks FOR SELECT USING (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='flashcard_decks' AND policyname='users can delete own decks') THEN
    CREATE POLICY "users can delete own decks" ON flashcard_decks FOR DELETE USING (auth.uid()::text = user_id);
  END IF;
END $$;


-- ----------------------------------------------------------------
-- flashcards (joined through flashcard_decks — allow any authenticated
-- user to insert/select cards whose deck belongs to them)
-- ----------------------------------------------------------------
ALTER TABLE flashcards ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='flashcards' AND policyname='users can insert cards for own decks') THEN
    CREATE POLICY "users can insert cards for own decks" ON flashcards FOR INSERT
      WITH CHECK (
        EXISTS (
          SELECT 1 FROM flashcard_decks d
          WHERE d.id = deck_id AND d.user_id = auth.uid()::text
        )
      );
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='flashcards' AND policyname='users can view cards for own decks') THEN
    CREATE POLICY "users can view cards for own decks" ON flashcards FOR SELECT
      USING (
        EXISTS (
          SELECT 1 FROM flashcard_decks d
          WHERE d.id = deck_id AND d.user_id = auth.uid()::text
        )
      );
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='flashcards' AND policyname='users can delete cards for own decks') THEN
    CREATE POLICY "users can delete cards for own decks" ON flashcards FOR DELETE
      USING (
        EXISTS (
          SELECT 1 FROM flashcard_decks d
          WHERE d.id = deck_id AND d.user_id = auth.uid()::text
        )
      );
  END IF;
END $$;


-- ----------------------------------------------------------------
-- quiz_sessions (user_id col)
-- ----------------------------------------------------------------
ALTER TABLE quiz_sessions ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='quiz_sessions' AND policyname='users can insert own quiz sessions') THEN
    CREATE POLICY "users can insert own quiz sessions" ON quiz_sessions FOR INSERT WITH CHECK (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='quiz_sessions' AND policyname='users can view own quiz sessions') THEN
    CREATE POLICY "users can view own quiz sessions" ON quiz_sessions FOR SELECT USING (auth.uid()::text = user_id);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='quiz_sessions' AND policyname='users can update own quiz sessions') THEN
    CREATE POLICY "users can update own quiz sessions" ON quiz_sessions FOR UPDATE USING (auth.uid()::text = user_id) WITH CHECK (auth.uid()::text = user_id);
  END IF;
END $$;


-- ----------------------------------------------------------------
-- session_topics (session_id only — no user_id; allow any auth user
-- to read/write their session row; the app uses session_id as the key)
-- ----------------------------------------------------------------
ALTER TABLE session_topics ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='session_topics' AND policyname='authenticated users can manage session topics') THEN
    CREATE POLICY "authenticated users can manage session topics" ON session_topics
      FOR ALL
      USING (auth.role() = 'authenticated')
      WITH CHECK (auth.role() = 'authenticated');
  END IF;
END $$;


-- ----------------------------------------------------------------
-- conversation_summaries (session_id only — same approach)
-- ----------------------------------------------------------------
ALTER TABLE conversation_summaries ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='conversation_summaries' AND policyname='authenticated users can manage conversation summaries') THEN
    CREATE POLICY "authenticated users can manage conversation summaries" ON conversation_summaries
      FOR ALL
      USING (auth.role() = 'authenticated')
      WITH CHECK (auth.role() = 'authenticated');
  END IF;
END $$;
