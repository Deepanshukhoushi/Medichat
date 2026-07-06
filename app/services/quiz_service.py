from __future__ import annotations

import json
import logging
import re

from app.core.security.exceptions import ServiceError
from app.repositories.quiz_repository import QuizRepository


logger = logging.getLogger(__name__)


class QuizService:
    def __init__(self, quiz_repository: QuizRepository | None = None) -> None:
        self.repository = quiz_repository

    def generate_quiz(self, user_id: str, topic: str, count: int, llm) -> str:
        """
        Generate a multiple-choice quiz of `count` questions for `topic`,
        save it, and return the session_id.
        """
        if not self.repository:
            raise ServiceError("Persistence is disabled")
            
        prompt = (
            f"Generate exactly {count} multiple-choice questions for medical students on the topic: '{topic}'.\n"
            "Return ONLY a JSON array of objects, with no markdown formatting and no extra text.\n"
            "Each object must have:\n"
            " - 'question': string\n"
            " - 'options': array of exactly 4 strings\n"
            " - 'correct': integer index (0-3) of the correct option\n"
            " - 'explanation': string explaining why it is correct"
        )
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            content = content.strip()
            content = re.sub(r"^```json", "", content, flags=re.IGNORECASE)
            content = re.sub(r"^```", "", content)
            content = re.sub(r"```$", "", content)
            content = content.strip()

            questions = json.loads(content)
            if not isinstance(questions, list) or len(questions) == 0:
                raise ValueError("Invalid format: expected a non-empty JSON array")
                
            return self.repository.save_session(user_id, topic, questions)
        except Exception as exc:
            logger.exception("Failed to generate quiz")
            raise ServiceError("Failed to generate quiz from AI") from exc

    def list_sessions(self, user_id: str) -> list[dict]:
        if not self.repository:
            return []
        return self.repository.list_sessions(user_id)

    def get_session(self, session_id: str, user_id: str) -> dict:
        if not self.repository:
            raise ServiceError("Persistence is disabled")
        return self.repository.get_session(session_id, user_id)

    def submit_score(self, session_id: str, user_id: str, score: int) -> None:
        if not self.repository:
            raise ServiceError("Persistence is disabled")
        self.repository.submit_score(session_id, user_id, score)

    def grade_answers(self, session_id: str, user_id: str, answers: list[int]) -> int:
        if not self.repository:
            raise ServiceError("Persistence is disabled")

        session = self.repository.get_session(session_id, user_id)
        questions = session.get("questions", [])
        if not isinstance(questions, list) or not questions:
            raise ServiceError("Quiz session is invalid")

        correct_count = 0
        for index, question in enumerate(questions):
            try:
                if int(answers[index]) == int(question.get("correct")):
                    correct_count += 1
            except (IndexError, TypeError, ValueError):
                continue

        score = round((correct_count / len(questions)) * 100)
        self.repository.submit_score(session_id, user_id, score)
        return score
