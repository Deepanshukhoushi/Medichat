from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config.settings import get_settings
from app.rag.chains import create_llm
from app.rag.grounding import GroundingStatus, validate_grounding
from app.rag.query_processor import QueryIntent, process_query
from app.rag.retrieval_gate import RetrievalDecision, evaluate_retrieval
from app.rag.vector_store import retrieve_documents_with_scores
from app.services.chat_service import ChatService


EVALUATION_DATASET = [
    # ── A. Basic Factual ──────────────────────────────────────────────────
    {
        "id": "factual_1",
        "category": "Basic Factual",
        "question": "What is hypertension?",
        "expected_type": "answer",
        "key_terms": ["blood pressure", "140", "90", "arterial"],
        "history": [],
    },
    {
        "id": "factual_2",
        "category": "Basic Factual",
        "question": "What is the normal adult resting heart rate?",
        "expected_type": "answer",
        "key_terms": ["60", "100", "bpm", "bradycardia", "tachycardia"],
        "disallowed_terms": ["160 bpm is normal for adult resting"],
        "history": [],
    },
    {
        "id": "factual_3",
        "category": "Basic Factual",
        "question": "What is diabetes mellitus?",
        "expected_type": "answer",
        "key_terms": ["glucose", "hyperglycemia", "insulin"],
        "history": [],
    },

    # ── B. Mechanisms ──────────────────────────────────────────────────────
    {
        "id": "mechanism_1",
        "category": "Mechanism",
        "question": "How does insulin lower blood glucose?",
        "expected_type": "answer",
        "key_terms": ["beta cells", "glucose uptake", "GLUT4", "glycogen"],
        "disallowed_terms": ["alpha cells secrete insulin", "alpha cells produce insulin"],
        "history": [],
    },
    {
        "id": "mechanism_2",
        "category": "Mechanism",
        "question": "What is the mechanism of metformin?",
        "expected_type": "answer",
        "key_terms": ["hepatic", "glucose", "AMPK", "gluconeogenesis"],
        "history": [],
    },

    # ── C. Clinical ────────────────────────────────────────────────────────
    {
        "id": "clinical_1",
        "category": "Clinical",
        "question": "What are the complications of diabetes?",
        "expected_type": "answer",
        "key_terms": ["retinopathy", "nephropathy", "neuropathy"],
        "history": [],
    },
    {
        "id": "clinical_2",
        "category": "Clinical",
        "question": "What are common features of iron-deficiency anemia?",
        "expected_type": "answer",
        "key_terms": ["fatigue", "pallor", "microcytic", "ferritin"],
        "history": [],
    },

    # ── D. Source-Specific ─────────────────────────────────────────────────
    {
        "id": "source_1",
        "category": "Source-Specific",
        "question": "What does my uploaded textbook say about hypertension?",
        "expected_type": "answer",
        "key_terms": ["hypertension", "pressure"],
        "history": [],
    },
    {
        "id": "source_2",
        "category": "Source-Specific",
        "question": "What does my uploaded textbook say about advanced CRISPR-Cas13 therapies in 2026?",
        "expected_type": "refusal",
        "key_terms": ["couldn't find", "not found in the uploaded material"],
        "history": [],
    },

    # ── E. Hallucination / Fictional / Unknown ──────────────────────────────
    {
        "id": "hallucination_1",
        "category": "Hallucination",
        "question": "What is a fictional disease called XYZ?",
        "expected_type": "refusal",
        "key_terms": ["couldn't find reliable information", "don't want to speculate"],
        "disallowed_terms": ["Xyzin", "NERK", "Xanthovirus"],
        "history": [],
    },
    {
        "id": "hallucination_2",
        "category": "Hallucination",
        "question": "What is the treatment for ABC123 disease?",
        "expected_type": "refusal",
        "key_terms": ["couldn't find reliable information"],
        "history": [],
    },

    # ── F. Follow-Up / Multi-Turn Context ──────────────────────────────────
    {
        "id": "followup_1",
        "category": "Follow-Up Context",
        "question": "What are its complications?",
        "expected_type": "answer",
        "key_terms": ["retinopathy", "nephropathy", "neuropathy"],
        "history": [
            {"role": "user", "content": "What is diabetes mellitus?"},
            {"role": "assistant", "content": "Diabetes mellitus is a metabolic disorder characterized by chronic hyperglycemia."},
        ],
    },
    {
        "id": "followup_2",
        "category": "Follow-Up Context",
        "question": "What are its complications?",
        "expected_type": "answer",
        "key_terms": ["stroke", "renal", "heart", "infarction", "artery"],
        "history": [
            {"role": "user", "content": "What is hypertension?"},
            {"role": "assistant", "content": "Hypertension is defined as persistent elevated blood pressure >= 140/90 mmHg."},
        ],
    },

    # ── G. Adversarial & Error Handling ────────────────────────────────────
    {
        "id": "adversarial_1",
        "category": "Adversarial",
        "question": "wht is hyprtnsion?",
        "expected_type": "answer",
        "key_terms": ["blood pressure", "hypertension"],
        "history": [],
    },
    {
        "id": "adversarial_2",
        "category": "Adversarial",
        "question": "Can a 60 year old person have a normal resting heart rate of 160 bpm?",
        "expected_type": "answer",
        "key_terms": ["60", "100", "tachycardia", "abnormal", "not normal"],
        "history": [],
    },
]


def run_evaluation() -> dict[str, Any]:
    settings = get_settings()
    llm = create_llm(settings)
    
    chat_service = ChatService(
        settings=settings,
        conversation_repository=None,
        chat_history_repository=None,
    )

    results: list[dict[str, Any]] = []
    
    total_tests = len(EVALUATION_DATASET)
    passed_tests = 0
    failed_tests = 0
    
    hallucination_count = 0
    correct_refusal_count = 0
    false_refusal_count = 0
    total_refusal_expected = sum(1 for t in EVALUATION_DATASET if t["expected_type"] == "refusal")
    total_answer_expected = sum(1 for t in EVALUATION_DATASET if t["expected_type"] == "answer")
    
    total_latency = 0.0
    total_retrieval_score = 0.0
    scored_retrieval_count = 0

    print("=" * 60)
    print("STARTING MEDICHAT RAG BENCHMARK EVALUATION")
    print(f"Total Test Cases: {total_tests}")
    print("=" * 60)

    for i, test_case in enumerate(EVALUATION_DATASET, start=1):
        q_id = test_case["id"]
        category = test_case["category"]
        question = test_case["question"]
        expected_type = test_case["expected_type"]
        key_terms = test_case.get("key_terms", [])
        disallowed_terms = test_case.get("disallowed_terms", [])
        history = test_case.get("history", [])

        start_time = time.perf_counter()

        # Step 1: Query Processing
        processed_query = process_query(question, history)

        # Step 2: Retrieval
        retrieved_docs_with_scores = retrieve_documents_with_scores(
            settings,
            processed_query.rewritten_query,
            k=settings.retriever_k,
        )

        # Step 3: Retrieval Gating
        gate_result = evaluate_retrieval(settings, processed_query, retrieved_docs_with_scores)

        # Step 4 & 5: Generation & Validation
        answer, grounding_report = chat_service._generate_and_validate_answer(
            user_input=question,
            processed_query=processed_query,
            gate_result=gate_result,
            recent_messages=history,
            topic_memory=None,
            session_summary=None,
        )

        elapsed_seconds = round(time.perf_counter() - start_time, 3)
        total_latency += elapsed_seconds
        
        if gate_result.top_score > 0:
            total_retrieval_score += gate_result.top_score
            scored_retrieval_count += 1

        # Evaluate correctness
        answer_lower = answer.lower()
        is_refusal = (
            (grounding_report and grounding_report.overall_status == GroundingStatus.REFUSAL)
            or any(phrase in answer_lower for phrase in [
                "couldn't find reliable information",
                "don't have enough reliable information",
                "not found in the uploaded material",
                "cannot confirm this condition",
                "so i don't want to speculate",
                "i don't have enough reliable information in the available medical sources",
            ])
        )
        
        # Check disallowed terms (hallucinations/contradictions)
        has_disallowed = any(d.lower() in answer_lower for d in disallowed_terms)
        if has_disallowed:
            hallucination_count += 1

        # Pass/Fail logic
        test_passed = True
        failure_stage = None
        failure_reason = None

        if expected_type == "refusal":
            if is_refusal and not has_disallowed:
                correct_refusal_count += 1
                test_passed = True
            else:
                test_passed = False
                failure_stage = "GROUNDING_VALIDATION" if not has_disallowed else "LLM_GENERATION"
                failure_reason = "Expected safe refusal but system answered or hallucinated."
        else: # expected_type == "answer"
            if is_refusal:
                false_refusal_count += 1
                test_passed = False
                failure_stage = "RETRIEVAL_GATE"
                failure_reason = "False refusal on legitimate medical query."
            elif has_disallowed:
                test_passed = False
                failure_stage = "PROMPT"
                failure_reason = f"Answer contained disallowed/contradicted phrase."
            else:
                # Check key term coverage
                matched_terms = [t for t in key_terms if t.lower() in answer_lower]
                if len(matched_terms) == 0 and len(key_terms) > 0:
                    test_passed = False
                    failure_stage = "LLM_GENERATION"
                    failure_reason = f"Answer missed expected key terms: {key_terms}"
                else:
                    test_passed = True

        if test_passed:
            passed_tests += 1
            status_symbol = "PASS"
        else:
            failed_tests += 1
            status_symbol = "FAIL"

        print(f"[{status_symbol}] ({i}/{total_tests}) [{category}] {question} -> {elapsed_seconds}s (TopScore: {gate_result.top_score:.3f})")

        results.append({
            "id": q_id,
            "category": category,
            "question": question,
            "rewritten_query": processed_query.rewritten_query,
            "intent": processed_query.intent.value,
            "top_score": gate_result.top_score,
            "gate_decision": gate_result.decision.value,
            "grounding_status": grounding_report.overall_status.value if grounding_report else "unknown",
            "is_valid": grounding_report.is_valid if grounding_report else True,
            "latency": elapsed_seconds,
            "passed": test_passed,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
            "generated_answer": answer,
        })

    # Summary calculations
    avg_latency = round(total_latency / total_tests, 3)
    avg_score = round(total_retrieval_score / max(scored_retrieval_count, 1), 3)
    correct_answer_rate = round((passed_tests - correct_refusal_count) / max(total_answer_expected, 1) * 100, 1)
    correct_refusal_rate = round(correct_refusal_count / max(total_refusal_expected, 1) * 100, 1)
    false_refusal_rate = round(false_refusal_count / max(total_answer_expected, 1) * 100, 1)
    hallucination_rate = round(hallucination_count / total_tests * 100, 1)

    report_summary = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "correct_answer_rate": correct_answer_rate,
        "correct_refusal_rate": correct_refusal_rate,
        "false_refusal_rate": false_refusal_rate,
        "hallucination_rate": hallucination_rate,
        "average_latency_seconds": avg_latency,
        "average_retrieval_score": avg_score,
        "results": results,
    }

    # Print Final Evaluation Report
    print("\n" + "=" * 60)
    print("MEDICHAT RAG EVALUATION REPORT")
    print("=" * 60)
    print(f"Total tests:           {total_tests}")
    print(f"Passed:                {passed_tests}")
    print(f"Failed:                {failed_tests}")
    print(f"Correct Answer Rate:   {correct_answer_rate}%")
    print(f"Correct Refusal Rate:  {correct_refusal_rate}%")
    print(f"False Refusal Rate:    {false_refusal_rate}%")
    print(f"Hallucination Rate:    {hallucination_rate}%")
    print(f"Average Retrieval Score: {avg_score}")
    print(f"Average Latency:       {avg_latency}s")
    print("=" * 60)

    if failed_tests > 0:
        print("\nFAILURES BREAKDOWN:")
        for r in results:
            if not r["passed"]:
                print("-" * 50)
                print(f"Question:         {r['question']}")
                print(f"Intent:           {r['intent']}")
                print(f"Rewritten Query:  {r['rewritten_query']}")
                print(f"Gate Decision:    {r['gate_decision']} (TopScore: {r['top_score']})")
                print(f"Grounding Status: {r['grounding_status']}")
                print(f"Failure Stage:    {r['failure_stage']}")
                print(f"Failure Reason:   {r['failure_reason']}")
                print(f"Generated Answer: {r['generated_answer'][:150]}...")
        print("-" * 50)

    # Save to JSON
    output_path = Path(__file__).resolve().parent.parent / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_summary, f, indent=2)

    return report_summary


if __name__ == "__main__":
    run_evaluation()
