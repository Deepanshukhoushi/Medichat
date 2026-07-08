from __future__ import annotations

import logging

from app.core.security.exceptions import RepositoryError


logger = logging.getLogger(__name__)


class FlashcardRepository:
    def __init__(self, supabase_client) -> None:
        self.supabase = supabase_client

    def save_deck(self, user_id: str, topic: str, cards: list[dict]) -> str:
        try:
            deck_res = self.supabase.table("flashcard_decks").insert({"user_id": user_id, "topic": topic}).execute()
            deck_id = deck_res.data[0]["id"]
            
            records = [{"deck_id": deck_id, "front": c["front"], "back": c["back"]} for c in cards]
            self.supabase.table("flashcards").insert(records).execute()
            return deck_id
        except Exception as exc:
            logger.exception("Failed to save flashcard deck")
            raise RepositoryError("Failed to save flashcard deck") from exc

    def list_decks(self, user_id: str, limit: int = 30) -> list[dict]:
        try:
            res = (
                self.supabase.table("flashcard_decks")
                .select("id, topic, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data
        except Exception as exc:
            logger.exception("Failed to list flashcard decks")
            raise RepositoryError("Failed to list flashcard decks") from exc

    def get_deck(self, deck_id: str, user_id: str) -> dict:
        try:
            deck_res = self.supabase.table("flashcard_decks").select("*").eq("id", deck_id).eq("user_id", user_id).execute()
            if not deck_res.data:
                raise RepositoryError("Deck not found")
            deck = deck_res.data[0]
            
            cards_res = self.supabase.table("flashcards").select("*").eq("deck_id", deck_id).execute()
            deck["cards"] = cards_res.data
            return deck
        except Exception as exc:
            logger.exception("Failed to get flashcard deck")
            raise RepositoryError("Failed to get flashcard deck") from exc

    def rate_card(self, deck_id: str, card_id: str, rating: str) -> None:
        try:
            card_res = self.supabase.table("flashcards").select("difficulty").eq("id", card_id).eq("deck_id", deck_id).execute()
            if not card_res.data:
                raise RepositoryError("Card not found")
            
            difficulty = card_res.data[0].get("difficulty", 3)
            if rating == "known":
                difficulty = max(1, difficulty - 1)
            elif rating == "unknown":
                difficulty = min(5, difficulty + 1)
                
            self.supabase.table("flashcards").update({"difficulty": difficulty}).eq("id", card_id).execute()
        except Exception as exc:
            logger.exception("Failed to rate flashcard")
            raise RepositoryError("Failed to rate flashcard") from exc
