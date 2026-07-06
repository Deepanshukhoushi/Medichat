from __future__ import annotations

import logging

from app.core.security.exceptions import ServiceError


logger = logging.getLogger(__name__)


class StudyToolsService:
    def explain(self, topic: str, llm) -> str:
        prompt = (
            f"Explain the medical topic '{topic}' clearly to a medical student.\n"
            "Use simple analogies where appropriate, but maintain clinical accuracy. "
            "Use headings and bullet points to structure the answer."
        )
        try:
            res = llm.invoke(prompt)
            return (res.content if hasattr(res, "content") else str(res)).strip()
        except Exception as exc:
            logger.exception("Explain generation failed")
            raise ServiceError("Failed to generate explanation") from exc

    def summarize(self, text: str, llm) -> str:
        prompt = (
            "Please provide a concise summary of the following medical text, "
            "highlighting the key clinical takeaways:\n\n"
            f"{text}"
        )
        try:
            res = llm.invoke(prompt)
            return (res.content if hasattr(res, "content") else str(res)).strip()
        except Exception as exc:
            logger.exception("Summarize generation failed")
            raise ServiceError("Failed to generate summary") from exc

    def generate_mnemonics(self, topic: str, llm) -> str:
        prompt = (
            f"Provide a few helpful, memorable mnemonics for learning about '{topic}'. "
            "If standard medical mnemonics exist, include them. Explain what each letter stands for clearly."
        )
        try:
            res = llm.invoke(prompt)
            return (res.content if hasattr(res, "content") else str(res)).strip()
        except Exception as exc:
            logger.exception("Mnemonics generation failed")
            raise ServiceError("Failed to generate mnemonics") from exc
