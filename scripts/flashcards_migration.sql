-- ============================================================
-- MediChat — Flashcards Migration
-- Run this script in your Supabase SQL editor.
-- ============================================================

CREATE TABLE IF NOT EXISTS flashcard_decks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    topic text NOT NULL,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_flashcard_decks_user
    ON flashcard_decks (user_id);

CREATE TABLE IF NOT EXISTS flashcards (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    deck_id uuid REFERENCES flashcard_decks(id) ON DELETE CASCADE,
    front text NOT NULL,
    back text NOT NULL,
    difficulty int DEFAULT 0 -- 0=new, 1=easy, 2=medium, 3=hard
);

CREATE INDEX IF NOT EXISTS idx_flashcards_deck
    ON flashcards (deck_id);
