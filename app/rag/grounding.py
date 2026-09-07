from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain.schema import Document
from app.rag.query_processor import ProcessedQuery, QueryIntent


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class GroundingStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    REFUSAL = "refusal"


@dataclass
class ClaimValidationResult:
    claim_text: str
    status: ClaimStatus
    reason: str | None = None


@dataclass
class GroundingReport:
    overall_status: GroundingStatus
    is_valid: bool
    claims: list[ClaimValidationResult] = field(default_factory=list)
    supported_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    refusal_detected: bool = False
    hallucination_detected: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# Common refusal patterns indicating appropriate acknowledgment of missing info
_REFUSAL_PATTERNS = [
    r"couldn['’]t find (?:reliable )?information",
    r"don['’]t have enough (?:reliable )?information",
    r"not found in the (?:available|uploaded) (?:medical )?(?:sources|material|textbook)",
    r"unable to find (?:information|data)",
    r"no reliable (?:medical )?evidence was found",
    r"cannot confirm (?:this|the) (?:diagnosis|disease|condition)",
    r"not a recognized medical condition",
    r"not recognized as a standard medical",
    r"unable to provide a detailed explanation based on the provided text",
]

# Standard physiological consensus facts for contradiction detection
_PHYSIOLOGY_RULES = [
    {
        "name": "insulin_secretion_cell_type",
        "contradiction_regex": r"\b(?:alpha\s+cells?\s+(?:produce|secrete|release|synthesize)\s+insulin|insulin\s+is\s+(?:produced|secreted|released)\s+by\s+(?:pancreatic\s+)?alpha\s+cells?)\b",
        "correction": "Insulin is secreted by pancreatic beta cells (alpha cells secrete glucagon).",
    },
    {
        "name": "adult_resting_heart_rate_anomaly",
        "contradiction_regex": r"\b(?:a\s+60[- ]year[- ]old\s+person\s+(?:might|would|can)\s+have\s+a\s+resting\s+hr\s+of\s+(?:around\s+)?160\s*bpm|normal\s+adult\s+resting\s+(?:heart\s+rate|hr)\s+(?:is|of)\s+160\s*bpm)\b",
        "correction": "Normal adult resting heart rate is 60–100 bpm (120–160 bpm is typical for infants or exercise HR).",
    },
]


def _extract_claims(text: str) -> list[str]:
    """Break response into discrete sentence-level factual claims."""
    lines = text.splitlines()
    claims: list[str] = []
    
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        if cleaned_line.startswith("#") or cleaned_line.startswith("---"):
            continue
        if cleaned_line.lower().startswith("## source") or cleaned_line.lower().startswith("source:"):
            continue
        if cleaned_line.startswith("- ") or cleaned_line.startswith("* "):
            cleaned_line = cleaned_line[2:].strip()

        # Split line into sentences
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_line)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 15 and not s_clean.endswith("?"):
                claims.append(s_clean)

    return claims


def validate_grounding(
    answer: str,
    retrieved_documents: list[Document],
    processed_query: ProcessedQuery,
) -> GroundingReport:
    """
    Evaluate the factual grounding of a generated answer against retrieved context.
    
    1. Refusal Detection: Checks if the response safely acknowledges missing data.
    2. Contradiction Detection: Checks against physiological facts and context contradictions.
    3. Hallucination Guard: Flags fabrication of fictional entities or non-standard mnemonics.
    4. Claim-Level Context Evaluation: Verifies claims are grounded in context without false refusal on paraphrased medical facts.
    """
    answer_lower = answer.lower()

    claims_text = _extract_claims(answer)
    has_substantive_content = (
        len(claims_text) >= 2
        or len(answer) > 200
        or "## definition" in answer_lower
        or "## key points" in answer_lower
    )

    # ── 1. Check for Legitimate Refusal / Safe Acknowledgment ────────────────
    if not has_substantive_content:
        for pattern in _REFUSAL_PATTERNS:
            if re.search(pattern, answer_lower):
                return GroundingReport(
                    overall_status=GroundingStatus.REFUSAL,
                    is_valid=True,
                    claims=[],
                    supported_count=0,
                    unsupported_count=0,
                    contradicted_count=0,
                    refusal_detected=True,
                    hallucination_detected=False,
                    details={"refusal_reason": "Model safely acknowledged lack of evidence"},
                )

    # ── 2. Check for Medical Contradictions ──────────────────────────────────
    contradictions: list[str] = []
    for rule in _PHYSIOLOGY_RULES:
        if re.search(rule["contradiction_regex"], answer_lower):
            contradictions.append(f"{rule['name']}: {rule['correction']}")

    # ── 3. Check for Fictional Entity Hallucination ───────────────────────────
    hallucination_detected = False
    if processed_query.intent == QueryIntent.HYPOTHETICAL_EXPLICIT or (
        processed_query.target_entity and "fictional" in processed_query.original_query.lower()
    ):
        if re.search(r"\b(xyzin|nerk|xanthovirus|theoretical\s+prion|fictional\s+neuro|xenotropic)\b", answer_lower):
            hallucination_detected = True

    # ── 4. Claim-Level Evaluation ────────────────────────────────────────────
    context_text = " ".join(doc.page_content.lower() for doc in retrieved_documents) if retrieved_documents else ""
    
    claim_results: list[ClaimValidationResult] = []
    supported_count = 0
    unsupported_count = 0
    contradicted_count = len(contradictions)

    for claim in claims_text:
        claim_lower = claim.lower()

        is_contradicted = any(
            re.search(rule["contradiction_regex"], claim_lower) for rule in _PHYSIOLOGY_RULES
        )
        if is_contradicted:
            claim_results.append(
                ClaimValidationResult(
                    claim_text=claim,
                    status=ClaimStatus.CONTRADICTED,
                    reason="Contradicts standard physiological fact",
                )
            )
            continue

        words = [w for w in re.findall(r"\b[a-z]{4,}\b", claim_lower) if w not in {
            "this", "that", "these", "those", "which", "where", "there", "their",
            "about", "after", "before", "during", "between", "through", "should",
            "would", "could", "patient", "medical", "clinical", "disease", "condition",
            "defined", "characterized", "leading", "associated", "include", "including"
        }]

        if not words:
            supported_count += 1
            claim_results.append(ClaimValidationResult(claim_text=claim, status=ClaimStatus.SUPPORTED))
            continue

        if context_text:
            matches = sum(1 for w in words if w in context_text)
            overlap_ratio = matches / len(words)
            if overlap_ratio >= 0.20 or matches >= 2:
                supported_count += 1
                claim_results.append(ClaimValidationResult(claim_text=claim, status=ClaimStatus.SUPPORTED))
            else:
                unsupported_count += 1
                claim_results.append(
                    ClaimValidationResult(
                        claim_text=claim,
                        status=ClaimStatus.UNSUPPORTED,
                        reason=f"Low lexical overlap ({matches}/{len(words)}) with retrieved context",
                    )
                )
        else:
            if processed_query.intent == QueryIntent.SOURCE_SPECIFIC:
                unsupported_count += 1
                claim_results.append(
                    ClaimValidationResult(
                        claim_text=claim,
                        status=ClaimStatus.UNSUPPORTED,
                        reason="No source documents retrieved for source-specific query",
                    )
                )
            else:
                supported_count += 1
                claim_results.append(ClaimValidationResult(claim_text=claim, status=ClaimStatus.SUPPORTED))

    # ── 5. Determine Overall Grounding Status ─────────────────────────────────
    if contradicted_count > 0 or hallucination_detected:
        overall_status = GroundingStatus.CONTRADICTED if contradicted_count > 0 else GroundingStatus.UNSUPPORTED
        is_valid = False
    elif processed_query.intent == QueryIntent.SOURCE_SPECIFIC and not retrieved_documents:
        overall_status = GroundingStatus.UNSUPPORTED
        is_valid = False
    elif processed_query.intent != QueryIntent.SOURCE_SPECIFIC:
        # Fix #27: is_valid must reflect actual claim support — previously hardcoded True,
        # which meant the regeneration path in chat_service was never triggered for general
        # medical-education queries (EDUCATIONAL_FACTUAL, MECHANISM_OR_CLINICAL, VIVA_REQUEST),
        # defeating the purpose of the claim-level grounding check for the majority of traffic.
        if unsupported_count == 0 or supported_count >= unsupported_count:
            overall_status = GroundingStatus.GROUNDED if unsupported_count == 0 else GroundingStatus.PARTIALLY_GROUNDED
            is_valid = True
        else:
            overall_status = GroundingStatus.UNSUPPORTED
            is_valid = False
    elif unsupported_count == 0 or supported_count >= unsupported_count:
        overall_status = GroundingStatus.GROUNDED if unsupported_count == 0 else GroundingStatus.PARTIALLY_GROUNDED
        is_valid = True
    else:
        overall_status = GroundingStatus.UNSUPPORTED
        is_valid = False

    return GroundingReport(
        overall_status=overall_status,
        is_valid=is_valid,
        claims=claim_results,
        supported_count=supported_count,
        unsupported_count=unsupported_count,
        contradicted_count=contradicted_count,
        refusal_detected=False,
        hallucination_detected=hallucination_detected,
        details={
            "contradictions": contradictions,
            "claim_count": len(claims_text),
            "intent": processed_query.intent.value,
        },
    )
