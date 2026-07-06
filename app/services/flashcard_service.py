from __future__ import annotations

import json
import logging
import re

from app.core.security.exceptions import ServiceError
from app.repositories.flashcard_repository import FlashcardRepository


logger = logging.getLogger(__name__)


class FlashcardService:
    def __init__(self, flashcard_repository: FlashcardRepository | None = None) -> None:
        self.repository = flashcard_repository

    def generate_deck(self, user_id: str, topic: str, count: int, llm) -> str:
        """
        Generate `count` flashcards for `topic` using `llm`, save to DB,
        and return the generated deck_id.
        """
        if not self.repository:
            raise ServiceError("Persistence is disabled")
        
        prompt = (
            f"Generate exactly {count} flashcards for medical students on the topic: '{topic}'.\n"
            "Return ONLY a JSON array of objects, with no markdown formatting and no extra text.\n"
            "Each object must have exactly two string keys: 'front' (the question/cue) and 'back' (the answer/explanation)."
        )
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Clean up potential markdown code block artifacts
            content = content.strip()
            content = re.sub(r"^```json", "", content, flags=re.IGNORECASE)
            content = re.sub(r"^```", "", content)
            content = re.sub(r"```$", "", content)
            content = content.strip()

            cards = json.loads(content)
            if not isinstance(cards, list) or len(cards) == 0:
                raise ValueError("Invalid format: expected a non-empty JSON array")
                
            return self.repository.save_deck(user_id, topic, cards)
        except Exception as exc:
            logger.exception("Failed to generate flashcard deck")
            raise ServiceError("Failed to generate flashcards from AI") from exc

    def list_decks(self, user_id: str) -> list[dict]:
        if not self.repository:
            return []
        return self.repository.list_decks(user_id)

    def get_deck(self, deck_id: str, user_id: str) -> dict:
        if not self.repository:
            raise ServiceError("Persistence is disabled")
        return self.repository.get_deck(deck_id, user_id)
