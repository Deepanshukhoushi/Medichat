from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryIntent(str, Enum):
    SOURCE_SPECIFIC = "source_specific"
    EDUCATIONAL_FACTUAL = "educational_factual"
    MECHANISM_OR_CLINICAL = "mechanism_or_clinical"
    VIVA_REQUEST = "viva_request"
    HYPOTHETICAL_EXPLICIT = "hypothetical_explicit"
    AMBIGUOUS_FOLLOWUP = "ambiguous_followup"


@dataclass
class ProcessedQuery:
    original_query: str
    rewritten_query: str
    intent: QueryIntent
    target_entity: str | None = None
    extracted_keywords: list[str] = field(default_factory=list)
    is_ambiguous: bool = False
    ambiguity_candidates: list[str] = field(default_factory=list)
    is_followup: bool = False


# Regex patterns for intent detection
_SOURCE_SPECIFIC_PATTERNS = [
    r"\b(?:what\s+does\s+(?:my|the)\s+(?:uploaded\s+)?(?:textbook|book|notes|pdf|document|material|source)\s+say\s+about)\b",
    r"\b(?:according\s+to\s+(?:my|the)\s+(?:uploaded\s+)?(?:textbook|book|notes|pdf|document|material|source))\b",
    r"\b(?:in\s+(?:my|the)\s+(?:uploaded\s+)?(?:textbook|book|notes|pdf|document|material|source))\b",
    r"\b(?:from\s+(?:my|the)\s+(?:uploaded\s+)?(?:textbook|book|notes|pdf|document|material|source))\b",
]

_VIVA_PATTERNS = [
    r"\b(?:ask\s+me\s+(?:\d+\s+)?viva\s+questions?)\b",
    r"\b(?:give\s+me\s+(?:\d+\s+)?viva\s+questions?)\b",
    r"\b(?:generate\s+(?:\d+\s+)?viva\s+questions?)\b",
    r"\b(?:viva\s+questions?\s+(?:on|about|for))\b",
    r"\b(?:test\s+my\s+knowledge\s+on\s+viva)\b",
]

_HYPOTHETICAL_PATTERNS = [
    r"\b(?:fictional\s+disease|fictional\s+condition|fictional\s+syndrome|fictional\s+drug)\b",
    r"\b(?:invent\s+a\s+(?:disease|condition|syndrome|drug|case))\b",
    r"\b(?:make\s+up\s+a\s+(?:disease|condition|syndrome|drug|case))\b",
    r"\b(?:hypothetical\s+disease|imaginary\s+disease)\b",
    r"\b(?:imagine\s+a\s+(?:fictional|hypothetical|non-existent))\b",
]

_FOLLOWUP_PRONOUN_PATTERNS = [
    r"\b(?:its|it|this|these|the|that)\s+(?:complications?|symptoms?|causes?|pathophysiology|treatment|management|mechanism|features?|diagnosis|prognosis|side\s+effects?|actions?|function|role)\b",
    r"\b(?:how\s+does\s+(?:it|this|that)\s+work)\b",
    r"\b(?:how\s+is\s+(?:it|this|that)\s+treated)\b",
    r"\b(?:what\s+causes\s+(?:it|this|that))\b",
    r"\b(?:what\s+are\s+(?:its|the)\s+(?:complications?|symptoms?|features?|causes?|side\s+effects?))\b",
    r"\b(?:what\s+about\s+(?:its|the)\s+treatment)\b",
    r"^(?:what\s+are\s+its\s+complications\??)$",
    r"^(?:how\s+does\s+it\s+work\??)$",
    r"^(?:what\s+causes\s+it\??)$",
    r"^(?:how\s+to\s+treat\s+it\??)$",
]

_MECHANISM_PATTERNS = [
    # Fix #29: was r"\b(?:how\s+does\s+.+\s+(?:work|...))\b" — unbounded greedy .+ caused
    # O(n²) backtracking on non-matching 4000-char inputs (one per chat turn).
    # Capped to .{1,80}? so worst-case is O(80²) ≈ constant from the app's perspective.
    r"\b(?:how\s+does\s+.{1,80}?\s+(?:work|lower|increase|regulate|inhibit|activate|cause|affect))\b",
    r"\b(?:mechanism\s+of\s+action|moa|pathophysiology|biochemical\s+mechanism|cellular\s+mechanism)\b",
    r"\b(?:mechanism\s+of)\b",
    r"\b(?:complications\s+of)\b",
    r"\b(?:adverse\s+effects\s+of|side\s+effects\s+of)\b",
]

_STOP_WORDS = {
    "what", "is", "a", "an", "the", "are", "of", "in", "to", "for", "and", "or",
    "how", "does", "do", "can", "could", "would", "should", "tell", "me", "about",
    "explain", "describe", "discuss", "give", "overview", "summary", "my", "your",
    "uploaded", "textbook", "notes", "book", "material", "pdf", "file", "its", "it",
    "wht", "wat", "wt", "hw", "pls", "plz", "ur", "u"
}


def _extract_recent_entities(chat_history: list[dict]) -> list[str]:
    """Extract candidate medical entities from recent user and assistant messages."""
    entities: list[str] = []
    seen: set[str] = set()

    for msg in reversed(chat_history[-6:]):
        content = msg.get("content", "").strip()
        role = msg.get("role", "user")

        # Extract entities from user questions
        if role == "user":
            # Match patterns like "What is X?", "Explain X", "What are complications of X?"
            match = re.search(
                r"(?:what\s+is|what\s+are|explain|describe|tell\s+me\s+about|mechanism\s+of|complications\s+of)\s+([A-Za-z0-9\s\-]+?)(?:\?|$|\.|\,)",
                content,
                re.IGNORECASE,
            )
            if match:
                candidate = match.group(1).strip()
                cleaned = re.sub(r"\b(the|a|an|its|this|that|these)\b", "", candidate, flags=re.IGNORECASE).strip()
                if cleaned and len(cleaned) > 2 and cleaned.lower() not in seen:
                    entities.append(cleaned)
                    seen.add(cleaned.lower())

        # Also extract primary medical headings or topics from assistant responses
        if role == "assistant":
            for line in content.splitlines()[:5]:
                if line.startswith("## ") or line.startswith("# "):
                    topic = line.replace("#", "").strip()
                    cleaned = re.sub(r"\b(Definition|Overview|Summary|Key Points)\b", "", topic, flags=re.IGNORECASE).strip()
                    if cleaned and len(cleaned) > 2 and cleaned.lower() not in seen:
                        entities.append(cleaned)
                        seen.add(cleaned.lower())

    return entities


def _extract_target_entity(query: str) -> str | None:
    """Extract the primary medical entity from a standalone query."""
    # Pattern: "What is [Entity]?", "Explain [Entity]", "Mechanism of [Entity]"
    match = re.search(
        r"(?:what\s+is\s+(?:a\s+|an\s+|the\s+)?|what\s+are\s+(?:the\s+)?|explain\s+(?:the\s+)?|mechanism\s+of\s+|complications\s+of\s+|features\s+of\s+|treatment\s+for\s+|what\s+does\s+.*say\s+about\s+)([A-Za-z0-9\s\-]+?)(?:\?|$|\.|\,)",
        query,
        re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        cleaned = re.sub(r"\b(the|a|an|its|this|that|fictional\s+disease|disease\s+called|syndrome\s+called)\b", "", candidate, flags=re.IGNORECASE).strip()
        if cleaned:
            return cleaned

    # Fallback: remove stop words and join remaining words
    words = [w for w in re.findall(r"\b[A-Za-z0-9\-]+\b", query) if w.lower() not in _STOP_WORDS]
    if words:
        return " ".join(words[:4])
    return None


def _extract_keywords(text: str) -> list[str]:
    """Extract significant keywords from a text query."""
    words = re.findall(r"\b[A-Za-z0-9\-]{3,}\b", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def process_query(
    user_query: str,
    chat_history: list[dict] | None = None,
    topic_memory: dict | None = None,
) -> ProcessedQuery:
    """
    Process, classify, and rewrite a user's query for high-precision RAG retrieval.
    
    1. Identifies query intent.
    2. Contextually resolves pronouns and elliptical follow-ups.
    3. Detects ambiguous multi-antecedent references.
    4. Extracts keywords and target entities.
    """
    cleaned_query = user_query.strip()
    chat_history = chat_history or []
    
    # ── 1. Check for Explicit Hypothetical / Fictional Intent ────────────────
    for pattern in _HYPOTHETICAL_PATTERNS:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            entity = _extract_target_entity(cleaned_query)
            return ProcessedQuery(
                original_query=cleaned_query,
                rewritten_query=cleaned_query,
                intent=QueryIntent.HYPOTHETICAL_EXPLICIT,
                target_entity=entity,
                extracted_keywords=_extract_keywords(cleaned_query),
                is_ambiguous=False,
            )

    # ── 2. Check for Viva Request Intent ─────────────────────────────────────
    for pattern in _VIVA_PATTERNS:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            entity = _extract_target_entity(cleaned_query)
            return ProcessedQuery(
                original_query=cleaned_query,
                rewritten_query=cleaned_query,
                intent=QueryIntent.VIVA_REQUEST,
                target_entity=entity,
                extracted_keywords=_extract_keywords(cleaned_query),
                is_ambiguous=False,
            )

    # ── 3. Check for Source-Specific Intent ──────────────────────────────────
    is_source_specific = False
    for pattern in _SOURCE_SPECIFIC_PATTERNS:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            is_source_specific = True
            break

    # ── 4. Check for Follow-Up References & Pronoun Resolution ───────────────
    is_followup = False
    for pattern in _FOLLOWUP_PRONOUN_PATTERNS:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            is_followup = True
            break

    # Also detect short query fragments that imply follow-up (e.g. "complications?", "treatment?")
    if len(cleaned_query.split()) <= 3 and not re.search(r"\b(what\s+is|what\s+are|define)\b", cleaned_query, re.IGNORECASE):
        if any(w in cleaned_query.lower() for w in ["complications", "treatment", "symptoms", "causes", "management", "mechanism", "diagnosis"]):
            is_followup = True

    rewritten_query = cleaned_query
    is_ambiguous = False
    ambiguity_candidates: list[str] = []
    target_entity: str | None = None

    if is_followup and chat_history:
        recent_entities = _extract_recent_entities(chat_history)
        
        # Check topic memory as well
        if topic_memory and topic_memory.get("current_topic"):
            current_topic = topic_memory["current_topic"]
            if current_topic not in recent_entities:
                recent_entities.insert(0, current_topic)

        if len(recent_entities) == 1:
            target_entity = recent_entities[0]
            # Contextual rewrite: replace pronoun or prepend antecedent
            if re.search(r"\b(its|it|this|these|the|that)\b", cleaned_query, re.IGNORECASE):
                rewritten_query = re.sub(
                    r"\b(its|it|this\s+disease|this\s+condition|this\s+drug|the\s+disease|the\s+drug)\b",
                    f"{target_entity}'s" if "its" in cleaned_query.lower() else target_entity,
                    cleaned_query,
                    flags=re.IGNORECASE,
                )
            else:
                rewritten_query = f"{cleaned_query} of {target_entity}"
        elif len(recent_entities) > 1:
            # If there are multiple candidates from the very latest turn, evaluate ambiguity
            latest_turn_entities = recent_entities[:2]
            target_entity = latest_turn_entities[0]
            # Primary candidate is the most recent
            rewritten_query = re.sub(
                r"\b(its|it|this\s+disease|this\s+condition|this\s+drug|the\s+disease|the\s+drug)\b",
                f"{target_entity}'s" if "its" in cleaned_query.lower() else target_entity,
                cleaned_query,
                flags=re.IGNORECASE,
            )
            # Check if ambiguous: multiple distinct active topics in immediate prior turn
            if len(latest_turn_entities) >= 2 and abs(len(latest_turn_entities[0]) - len(latest_turn_entities[1])) < 20:
                is_ambiguous = False  # Prefer latest turn antecedent by default, record candidates
                ambiguity_candidates = latest_turn_entities
    else:
        target_entity = _extract_target_entity(cleaned_query)

    # ── 5. Determine Intent ──────────────────────────────────────────────────
    if is_source_specific:
        intent = QueryIntent.SOURCE_SPECIFIC
    else:
        # Fix #29: _MECHANISM_PATTERNS[0] is the bounded "how does ... work" regex.
        # Run a fast O(n) substring check first so the regex is only evaluated when
        # the query actually starts with "how does" (avoids even compiling the RE
        # match object for the vast majority of queries).
        query_lower = rewritten_query.lower()
        p0_match = (
            "how does" in query_lower
            and re.search(_MECHANISM_PATTERNS[0], rewritten_query, re.IGNORECASE)
        )
        mechanism_match = p0_match or any(
            re.search(p, rewritten_query, re.IGNORECASE) for p in _MECHANISM_PATTERNS[1:]
        )
        if mechanism_match:
            intent = QueryIntent.MECHANISM_OR_CLINICAL
        else:
            intent = QueryIntent.EDUCATIONAL_FACTUAL

    return ProcessedQuery(
        original_query=cleaned_query,
        rewritten_query=rewritten_query,
        intent=intent,
        target_entity=target_entity,
        extracted_keywords=_extract_keywords(rewritten_query),
        is_ambiguous=is_ambiguous,
        ambiguity_candidates=ambiguity_candidates,
        is_followup=is_followup,
    )
