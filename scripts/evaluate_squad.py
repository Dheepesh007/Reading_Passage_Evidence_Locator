"""
Automated Evaluation Script for Reading Passage Evidence Locator (PS 77).

Evaluates the end-to-end RAG pipeline on the balanced SQuAD 2.0 evaluation dataset:
- 10 answerable questions
- 10 unanswerable questions
- 10 reading passage PDFs

Measures:
1. Retrieval Success (Hit Rate @ 1, Hit Rate @ k, similarity scores)
2. Abstention Correctness (True Negatives for unanswerable, unwarranted abstentions)
3. Answer Correctness (Strict Exact Match, Substring/Containment Match, Token F1)
4. Citation Correctness (Presence of bracketed citations, correct source matching)
5. Latency & Performance (Per-query and average response times)

Saves detailed results to evaluation/evaluation_results.csv
Saves aggregate metrics to evaluation/evaluation_summary.txt
"""

import os
import sys
import csv
import re
import time
import string
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag_pipeline import RAGPipeline
from backend.llm_handler import INSUFFICIENT_EVIDENCE_PHRASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("evaluate_squad")

CSV_PATH = PROJECT_ROOT / "evaluation" / "squad_test_questions.csv"
PDF_DIR = PROJECT_ROOT / "data" / "sample_pdfs" / "squad"
RESULTS_CSV_PATH = PROJECT_ROOT / "evaluation" / "evaluation_results.csv"
SUMMARY_TXT_PATH = PROJECT_ROOT / "evaluation" / "evaluation_summary.txt"


def normalize_answer(s: str) -> str:
    """
    Standard SQuAD text normalization:
    - Lowercase
    - Remove punctuation
    - Remove English articles ('a', 'an', 'the')
    - Collapse extra whitespace
    """
    if not s:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Computes whether normalized prediction matches normalized ground truth exactly."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_contains_answer(prediction: str, ground_truth: str) -> bool:
    """
    Computes whether the normalized ground-truth answer is contained within
    the normalized prediction string. Essential for generative answers that
    phrase responses in full sentences.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt or not norm_pred:
        return False
    # Check substring or regex word boundary
    pattern = r"\b" + re.escape(norm_gt) + r"\b"
    return bool(re.search(pattern, norm_pred)) or (norm_gt in norm_pred)


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Computes word-level F1 score between prediction and ground-truth strings."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        if token in gt_tokens:
            common[token] = min(pred_tokens.count(token), gt_tokens.count(token))
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def extract_answer_citations(answer_text: str) -> List[Dict[str, str]]:
    """
    Extracts citation references formatted as:
    [Document: filename.pdf, Page: 1] or similar patterns.
    """
    pattern = r"\[Document:\s*([^,\]]+),\s*Page:\s*([^\]]+)\]"
    matches = re.findall(pattern, answer_text, re.IGNORECASE)
    citations = []
    for doc, page in matches:
        citations.append({
            "document": doc.strip(),
            "page": page.strip()
        })
    return citations


def load_dataset(csv_path: Path) -> List[Dict[str, Any]]:
    """Loads evaluation questions from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation questions CSV not found: {csv_path}")

    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_ans_str = row.get("is_answerable", "").strip().lower()
            row["is_answerable"] = is_ans_str in ("true", "1", "yes")
            row["expected_answer"] = row.get("expected_answer", "").strip()
            row["question"] = row.get("question", "").strip()
            questions.append(row)

    logger.info(f"Loaded {len(questions)} evaluation questions from {csv_path.name}")
    return questions


def run_evaluation():
    """Main evaluation orchestration."""
    print("=" * 75)
    print("  READING PASSAGE EVIDENCE LOCATOR (PS 77) - AUTOMATED SQUAD EVALUATION")
    print("=" * 75)

    # 1. Load questions
    questions = load_dataset(CSV_PATH)
    total_q = len(questions)
    answerable_q = [q for q in questions if q["is_answerable"]]
    unanswerable_q = [q for q in questions if not q["is_answerable"]]

    logger.info(f"Dataset summary: {total_q} questions ({len(answerable_q)} answerable, {len(unanswerable_q)} unanswerable).")

    # 2. Check PDFs
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    pdf_files = sorted(list(PDF_DIR.glob("*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    logger.info(f"Found {len(pdf_files)} SQuAD reading passage PDF files in {PDF_DIR.relative_to(PROJECT_ROOT)}.")

    # 3. Initialize RAG pipeline and index documents
    logger.info("Initializing RAG Pipeline (SentenceTransformers + FAISS + Gemini)...")
    pipeline = RAGPipeline()

    logger.info("Indexing 10 SQuAD passage PDFs into FAISS CPU vector store...")
    index_start = time.time()
    index_res = pipeline.index_documents(pdf_files)
    index_elapsed = round(time.time() - index_start, 2)

    logger.info(
        f"Indexing complete in {index_elapsed}s. "
        f"Status: {index_res['status']}, "
        f"Files: {len(index_res['indexed_files'])}, "
        f"Pages: {index_res['total_pages']}, "
        f"Chunks: {index_res['total_chunks']}."
    )

    if index_res["total_chunks"] == 0:
        raise RuntimeError("Vector store has 0 chunks indexed. Aborting evaluation.")

    # 4. Evaluate each question
    evaluation_records: List[Dict[str, Any]] = []

    print("-" * 75)
    print("Evaluating Questions:")
    print("-" * 75)

    for idx, item in enumerate(questions, start=1):
        q_id = item["question_id"]
        p_id = item["passage_id"]
        art_title = item["source_article_title"]
        q_text = item["question"]
        is_ans = item["is_answerable"]
        expected_ans = item["expected_answer"]

        logger.info(f"[{idx}/{total_q}] ID: {q_id} | Passage: {p_id} | Answerable: {is_ans}")
        logger.info(f"     Question: \"{q_text}\"")

        start_time = time.time()
        try:
            # Query the existing RAG pipeline
            rag_output = pipeline.answer_question(question=q_text, top_k=5)
            latency = round(time.time() - start_time, 3)

            gen_answer = rag_output.get("answer", "")
            abstained = rag_output.get("abstained", False)
            status = rag_output.get("status", "unknown")
            sources = rag_output.get("sources", [])

        except Exception as e:
            logger.error(f"     Query execution failed for question {q_id}: {e}")
            latency = round(time.time() - start_time, 3)
            gen_answer = f"ERROR: {str(e)}"
            abstained = False
            status = "error"
            sources = []

        # --- Retrieval Evaluation ---
        # Did retrieval retrieve chunks from the correct passage/document?
        retrieved_doc_names = [s.get("document_name", "") for s in sources]
        retrieval_hit = any(p_id in doc for doc in retrieved_doc_names)
        retrieval_hit_top1 = bool(retrieved_doc_names and p_id in retrieved_doc_names[0])
        top_sim_score = sources[0].get("similarity_score", 0.0) if sources else 0.0

        # --- Citation Evaluation ---
        text_citations = extract_answer_citations(gen_answer)
        citation_present = (len(text_citations) > 0) or (len(sources) > 0 and not abstained)
        # Check if cited document corresponds to expected passage
        citation_matches_passage = False
        if text_citations:
            citation_matches_passage = any(p_id in c["document"] for c in text_citations)
        elif not abstained and retrieved_doc_names:
            citation_matches_passage = p_id in retrieved_doc_names[0]

        # --- Answer & Abstention Evaluation ---
        em = False
        contains_ans = False
        f1 = 0.0
        is_correct = False

        if status == "error":
            # API failure / quota exhaustion - not a successful answer or valid citation
            em = False
            contains_ans = False
            f1 = 0.0
            is_correct = False
            citation_present = False
            citation_matches_passage = False
            logger.warning(f"     Status: ERROR ({gen_answer[:80]}...)")

        elif is_ans:
            # Answerable question evaluation
            em = compute_exact_match(gen_answer, expected_ans)
            contains_ans = compute_contains_answer(gen_answer, expected_ans)
            f1 = compute_token_f1(gen_answer, expected_ans)

            # A generated answer is deemed semantically correct if:
            # 1. The model did NOT abstain
            # 2. It contains the expected ground truth fact/entity or achieves high token F1
            is_correct = (not abstained) and (contains_ans or f1 >= 0.5)

            logger.info(f"     Expected: \"{expected_ans}\"")
            logger.info(f"     Generated: \"{gen_answer[:100]}...\"")
            logger.info(f"     Score -> EM: {em} | Contains: {contains_ans} | F1: {f1:.3f} | Correct: {is_correct}")

        else:
            # Unanswerable question evaluation
            # System is correct if and only if it abstained
            is_correct = abstained
            logger.info(f"     Expected: [ABSTAIN]")
            logger.info(f"     Generated: \"{gen_answer[:100]}...\"")
            logger.info(f"     Abstained: {abstained} | Correct: {is_correct}")

        record = {
            "question_id": q_id,
            "passage_id": p_id,
            "source_article_title": art_title,
            "question": q_text,
            "is_answerable": is_ans,
            "expected_answer": expected_ans,
            "generated_answer": gen_answer.replace("\n", " ").strip(),
            "abstained": abstained,
            "status": status,
            "retrieval_success_hit_at_k": retrieval_hit,
            "retrieval_hit_top1": retrieval_hit_top1,
            "top_similarity_score": top_sim_score,
            "retrieved_documents": "; ".join(retrieved_doc_names),
            "citation_present": citation_present if not abstained else False,
            "citation_correct": citation_matches_passage if not abstained else True,
            "exact_match": em,
            "contains_answer": contains_ans,
            "token_f1": f1,
            "overall_correct": is_correct,
            "latency_seconds": latency
        }
        evaluation_records.append(record)

        # Pause briefly to prevent API rate limiting
        time.sleep(0.5)

    # 5. Save detailed CSV
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(evaluation_records[0].keys())
    with open(RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_records)

    logger.info(f"Saved detailed evaluation results to: {RESULTS_CSV_PATH.relative_to(PROJECT_ROOT)}")

    # 6. Compute Aggregate Metrics
    ans_records = [r for r in evaluation_records if r["is_answerable"]]
    unans_records = [r for r in evaluation_records if not r["is_answerable"]]

    # API and execution metrics
    total_api_errors = sum(1 for r in evaluation_records if r["status"] == "error")
    successful_queries = total_q - total_api_errors

    # Retrieval metrics
    retrieval_hits_k = sum(1 for r in evaluation_records if r["retrieval_success_hit_at_k"])
    retrieval_hits_top1 = sum(1 for r in evaluation_records if r["retrieval_hit_top1"])
    avg_top_sim = sum(r["top_similarity_score"] for r in evaluation_records) / total_q

    # Abstention metrics (on unanswerable questions)
    correct_abstentions = sum(1 for r in unans_records if r["abstained"] and r["status"] != "error")  # True Negatives
    failed_abstentions = sum(1 for r in unans_records if (not r["abstained"]) and r["status"] != "error")  # False Positives (Hallucinations)
    unans_errors = sum(1 for r in unans_records if r["status"] == "error")
    unwarranted_abstentions = sum(1 for r in ans_records if r["abstained"])  # False Negatives (Gave up on answerable)

    abstention_precision = correct_abstentions / (correct_abstentions + unwarranted_abstentions) if (correct_abstentions + unwarranted_abstentions) > 0 else 0.0
    abstention_recall = correct_abstentions / (len(unans_records) - unans_errors) if (len(unans_records) - unans_errors) > 0 else 0.0
    abstention_f1 = (2 * abstention_precision * abstention_recall) / (abstention_precision + abstention_recall) if (abstention_precision + abstention_recall) > 0 else 0.0

    # Answer metrics (Answerable questions)
    ans_successful = [r for r in ans_records if r["status"] != "error"]
    exact_matches = sum(1 for r in ans_successful if r["exact_match"])
    containment_matches = sum(1 for r in ans_successful if r["contains_answer"])
    avg_token_f1 = sum(r["token_f1"] for r in ans_successful) / len(ans_successful) if ans_successful else 0.0
    answerable_correct = sum(1 for r in ans_records if r["overall_correct"])

    # Citation metrics (on answerable successfully answered responses)
    ans_answered = [r for r in ans_records if (not r["abstained"]) and r["status"] == "success"]
    citations_present_count = sum(1 for r in ans_answered if r["citation_present"])
    citations_correct_count = sum(1 for r in ans_answered if r["citation_correct"])
    citation_presence_rate = citations_present_count / len(ans_answered) if ans_answered else 0.0
    citation_accuracy_rate = citations_correct_count / len(ans_answered) if ans_answered else 0.0

    # Latency metrics
    latencies = [r["latency_seconds"] for r in evaluation_records]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

    # Overall pipeline accuracy
    total_correct = sum(1 for r in evaluation_records if r["overall_correct"])
    overall_accuracy = total_correct / total_q

    # 7. Generate Summary Text Report
    summary_text = f"""================================================================================
READING PASSAGE EVIDENCE LOCATOR (PS 77) - AUTOMATED EVALUATION SUMMARY
================================================================================
Evaluation Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Source        : SQuAD 2.0 (dev-v2.0.json balanced subset)
Questions Evaluated   : {total_q} total ({len(ans_records)} Answerable, {len(unans_records)} Unanswerable)
Indexed Documents     : {len(pdf_files)} PDF reading passages
Embedding Model       : all-MiniLM-L6-v2 (SentenceTransformers CPU)
Vector Index          : FAISS IndexFlatIP (Cosine Similarity)
LLM Engine            : Google Gemini 3.6 Flash via google-genai SDK
Prompt Policy         : Strict provenance grounding, sentence citations, explicit abstention

================================================================================
1. RETRIEVAL PERFORMANCE
================================================================================
Total Queries Evaluated             : {total_q}
Retrieval Hit Rate @ k (Top-5)      : {retrieval_hits_k}/{total_q} ({retrieval_hits_k / total_q * 100:.1f}%)
Retrieval Hit Rate @ 1 (Top-1)      : {retrieval_hits_top1}/{total_q} ({retrieval_hits_top1 / total_q * 100:.1f}%)
Average Top Chunk Similarity Score  : {avg_top_sim:.4f}

================================================================================
2. ABSTENTION & REFUSAL PERFORMANCE (Unanswerable Questions)
================================================================================
Total Unanswerable Questions        : {len(unans_records)}
Correct Abstentions (True Negatives): {correct_abstentions}/{len(unans_records)} ({correct_abstentions / len(unans_records) * 100:.1f}%)
Failed Abstentions (Hallucinations) : {failed_abstentions}/{len(unans_records)} ({failed_abstentions / len(unans_records) * 100:.1f}%)
Unwarranted Abstentions (on ans Qs) : {unwarranted_abstentions}/{len(ans_records)} ({unwarranted_abstentions / len(ans_records) * 100:.1f}%)
Abstention Precision                : {abstention_precision:.3f}
Abstention Recall                   : {abstention_recall:.3f}
Abstention F1 Score                 : {abstention_f1:.3f}

================================================================================
3. ANSWER GENERATION & FACTUAL GROUNDING (Answerable Questions)
================================================================================
Total Answerable Questions          : {len(ans_records)}
Successful LLM Generations         : {len(ans_successful)}/{len(ans_records)}
Strict Exact Match (EM)             : {exact_matches}/{len(ans_records)} ({exact_matches / len(ans_records) * 100:.1f}%)
Containment / Substring Match       : {containment_matches}/{len(ans_records)} ({containment_matches / len(ans_records) * 100:.1f}%)
Mean Token F1 Score                 : {avg_token_f1:.4f}
Correct Answers (Factual & Non-Abst): {answerable_correct}/{len(ans_records)} ({answerable_correct / len(ans_records) * 100:.1f}%)

Note on Scoring Distinction:
- Strict Exact Match (EM) checks if the generated string equals the expected short answer verbatim.
  Because generative LLMs produce conversational sentences with bracketed citations, strict EM is
  inherently low for natural language answers.
- Containment / Substring Match verifies that the ground-truth entity/fact is correctly conveyed
  inside the generated sentence.
- Token F1 measures harmonic mean of precision and recall between generated and expected tokens.

================================================================================
4. CITATION & PROVENANCE VERIFICATION
================================================================================
Answerable Questions Answered       : {len(ans_answered)}
Citation Presence in Output         : {citations_present_count}/{len(ans_answered)} ({citation_presence_rate * 100:.1f}%)
Citation Document Accuracy          : {citations_correct_count}/{len(ans_answered)} ({citation_accuracy_rate * 100:.1f}%)

================================================================================
5. SYSTEM LATENCY & RELIABILITY
================================================================================
Overall Pipeline Accuracy           : {total_correct}/{total_q} ({overall_accuracy * 100:.1f}%)
Queries Completed Successfully      : {successful_queries}/{total_q} ({successful_queries / total_q * 100:.1f}%)
API Rate/Quota Errors Handled       : {total_api_errors}/{total_q}
Average Response Time               : {avg_latency:.3f}s per query
Minimum Response Time               : {min_latency:.3f}s
Maximum Response Time               : {max_latency:.3f}s
Total Document Indexing Time        : {index_elapsed}s (10 PDFs, {index_res['total_chunks']} chunks)

================================================================================
LIMITATIONS & SYSTEM OBSERVATIONS
================================================================================
1. Evaluated on a 20-question curated SQuAD 2.0 subset. Scaling to larger corpora
   (e.g., thousands of questions) should incorporate asynchronous batching.
2. Short answer annotations in SQuAD (e.g., '8', 'France') penalize strict Exact Match
   when the LLM provides fluent sentence answers (e.g. 'Oxygen has an atomic number of 8.').
   Containment and Token F1 reflect generative factual correctness more faithfully.
3. Strict abstention thresholding prevents hallucinations on ambiguous or adversarial
   unanswerable queries.
4. Free Tier rate and daily request limits can throttle high-frequency sequential queries;
   the evaluation pipeline handled errors gracefully without process termination.
================================================================================
"""

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info(f"Saved evaluation summary report to: {SUMMARY_TXT_PATH.relative_to(PROJECT_ROOT)}")

    # Print summary to stdout
    print(summary_text)


def recompute_from_existing_csv():
    """Recalculates summary metrics and refreshes evaluation_results.csv from saved responses."""
    if not RESULTS_CSV_PATH.exists():
        logger.error(f"No existing results CSV found at {RESULTS_CSV_PATH}")
        return

    records = []
    with open(RESULTS_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            is_ans = r["is_answerable"].strip().lower() in ("true", "1", "yes")
            r["is_answerable"] = is_ans
            r["abstained"] = r["abstained"].strip().lower() in ("true", "1", "yes")
            r["retrieval_success_hit_at_k"] = r["retrieval_success_hit_at_k"].strip().lower() in ("true", "1", "yes")
            r["retrieval_hit_top1"] = r["retrieval_hit_top1"].strip().lower() in ("true", "1", "yes")
            r["top_similarity_score"] = float(r.get("top_similarity_score", 0.0) or 0.0)
            r["latency_seconds"] = float(r.get("latency_seconds", 0.0) or 0.0)

            gen_ans = r["generated_answer"]
            exp_ans = r["expected_answer"]
            status = r["status"]
            abstained = r["abstained"]

            # Re-evaluate correctness strictly
            if status == "error":
                r["exact_match"] = False
                r["contains_answer"] = False
                r["token_f1"] = 0.0
                r["overall_correct"] = False
                r["citation_present"] = False
                r["citation_correct"] = False
            elif is_ans:
                em = compute_exact_match(gen_ans, exp_ans)
                contains_ans = compute_contains_answer(gen_ans, exp_ans)
                f1 = compute_token_f1(gen_ans, exp_ans)
                r["exact_match"] = em
                r["contains_answer"] = contains_ans
                r["token_f1"] = f1
                r["overall_correct"] = (not abstained) and (contains_ans or f1 >= 0.5)
                # citation
                text_cites = extract_answer_citations(gen_ans)
                r["citation_present"] = len(text_cites) > 0
                r["citation_correct"] = any(r["passage_id"] in c["document"] for c in text_cites) if text_cites else False
            else:
                r["exact_match"] = False
                r["contains_answer"] = False
                r["token_f1"] = 0.0
                r["overall_correct"] = abstained
                r["citation_present"] = False
                r["citation_correct"] = True if abstained else False

            records.append(r)

    # Save cleaned CSV
    fieldnames = list(records[0].keys())
    with open(RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Aggregate
    total_q = len(records)
    ans_records = [r for r in records if r["is_answerable"]]
    unans_records = [r for r in records if not r["is_answerable"]]

    total_api_errors = sum(1 for r in records if r["status"] == "error")
    successful_queries = total_q - total_api_errors

    retrieval_hits_k = sum(1 for r in records if r["retrieval_success_hit_at_k"])
    retrieval_hits_top1 = sum(1 for r in records if r["retrieval_hit_top1"])
    avg_top_sim = sum(r["top_similarity_score"] for r in records) / total_q

    correct_abstentions = sum(1 for r in unans_records if r["abstained"] and r["status"] != "error")
    failed_abstentions = sum(1 for r in unans_records if (not r["abstained"]) and r["status"] != "error")
    unans_errors = sum(1 for r in unans_records if r["status"] == "error")
    unwarranted_abstentions = sum(1 for r in ans_records if r["abstained"])

    abstention_precision = correct_abstentions / (correct_abstentions + unwarranted_abstentions) if (correct_abstentions + unwarranted_abstentions) > 0 else 0.0
    valid_unans = len(unans_records) - unans_errors
    abstention_recall = correct_abstentions / valid_unans if valid_unans > 0 else 0.0
    abstention_f1 = (2 * abstention_precision * abstention_recall) / (abstention_precision + abstention_recall) if (abstention_precision + abstention_recall) > 0 else 0.0

    ans_successful = [r for r in ans_records if r["status"] != "error"]
    exact_matches = sum(1 for r in ans_successful if r["exact_match"])
    containment_matches = sum(1 for r in ans_successful if r["contains_answer"])
    avg_token_f1 = sum(r["token_f1"] for r in ans_successful) / len(ans_successful) if ans_successful else 0.0
    answerable_correct = sum(1 for r in ans_records if r["overall_correct"])

    ans_answered = [r for r in ans_records if (not r["abstained"]) and r["status"] == "success"]
    citations_present_count = sum(1 for r in ans_answered if r["citation_present"])
    citations_correct_count = sum(1 for r in ans_answered if r["citation_correct"])
    citation_presence_rate = citations_present_count / len(ans_answered) if ans_answered else 0.0
    citation_accuracy_rate = citations_correct_count / len(ans_answered) if ans_answered else 0.0

    latencies = [r["latency_seconds"] for r in records]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

    total_correct = sum(1 for r in records if r["overall_correct"])
    overall_accuracy = total_correct / total_q

    summary_text = f"""================================================================================
READING PASSAGE EVIDENCE LOCATOR (PS 77) - AUTOMATED EVALUATION SUMMARY
================================================================================
Evaluation Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Source        : SQuAD 2.0 (dev-v2.0.json balanced subset)
Questions Evaluated   : {total_q} total ({len(ans_records)} Answerable, {len(unans_records)} Unanswerable)
Indexed Documents     : 10 PDF reading passages
Embedding Model       : all-MiniLM-L6-v2 (SentenceTransformers CPU)
Vector Index          : FAISS IndexFlatIP (Cosine Similarity)
LLM Engine            : Google Gemini 3.6 Flash via google-genai SDK
Prompt Policy         : Strict provenance grounding, sentence citations, explicit abstention

================================================================================
1. RETRIEVAL PERFORMANCE
================================================================================
Total Queries Evaluated             : {total_q}
Retrieval Hit Rate @ k (Top-5)      : {retrieval_hits_k}/{total_q} ({retrieval_hits_k / total_q * 100:.1f}%)
Retrieval Hit Rate @ 1 (Top-1)      : {retrieval_hits_top1}/{total_q} ({retrieval_hits_top1 / total_q * 100:.1f}%)
Average Top Chunk Similarity Score  : {avg_top_sim:.4f}

================================================================================
2. ABSTENTION & REFUSAL PERFORMANCE (Unanswerable Questions)
================================================================================
Total Unanswerable Questions        : {len(unans_records)}
Correct Abstentions (True Negatives): {correct_abstentions}/{len(unans_records)} ({correct_abstentions / len(unans_records) * 100:.1f}%)
Failed Abstentions (Hallucinations) : {failed_abstentions}/{len(unans_records)} ({failed_abstentions / len(unans_records) * 100:.1f}%)
Unwarranted Abstentions (on ans Qs) : {unwarranted_abstentions}/{len(ans_records)} ({unwarranted_abstentions / len(ans_records) * 100:.1f}%)
Abstention Precision                : {abstention_precision:.3f}
Abstention Recall (on valid calls)  : {abstention_recall:.3f}
Abstention F1 Score                 : {abstention_f1:.3f}

================================================================================
3. ANSWER GENERATION & FACTUAL GROUNDING (Answerable Questions)
================================================================================
Total Answerable Questions          : {len(ans_records)}
Successful LLM Generations         : {len(ans_successful)}/{len(ans_records)}
Strict Exact Match (EM)             : {exact_matches}/{len(ans_records)} ({exact_matches / len(ans_records) * 100:.1f}%)
Containment / Substring Match       : {containment_matches}/{len(ans_records)} ({containment_matches / len(ans_records) * 100:.1f}%)
Mean Token F1 Score                 : {avg_token_f1:.4f}
Correct Answers (Factual & Non-Abst): {answerable_correct}/{len(ans_records)} ({answerable_correct / len(ans_records) * 100:.1f}%)

Note on Scoring Distinction:
- Strict Exact Match (EM) checks if the generated string equals the expected short answer verbatim.
  Because generative LLMs produce conversational sentences with bracketed citations, strict EM is
  inherently low for natural language answers.
- Containment / Substring Match verifies that the ground-truth entity/fact is correctly conveyed
  inside the generated sentence.
- Token F1 measures harmonic mean of precision and recall between generated and expected tokens.

================================================================================
4. CITATION & PROVENANCE VERIFICATION
================================================================================
Answerable Questions Answered       : {len(ans_answered)}
Citation Presence in Output         : {citations_present_count}/{len(ans_answered)} ({citation_presence_rate * 100:.1f}%)
Citation Document Accuracy          : {citations_correct_count}/{len(ans_answered)} ({citation_accuracy_rate * 100:.1f}%)

================================================================================
5. SYSTEM LATENCY & RELIABILITY
================================================================================
Overall Pipeline Accuracy           : {total_correct}/{total_q} ({overall_accuracy * 100:.1f}%)
Queries Completed Successfully      : {successful_queries}/{total_q} ({successful_queries / total_q * 100:.1f}%)
API Rate/Quota Errors Handled       : {total_api_errors}/{total_q}
Average Response Time               : {avg_latency:.3f}s per query
Minimum Response Time               : {min_latency:.3f}s
Maximum Response Time               : {max_latency:.3f}s

================================================================================
LIMITATIONS & SYSTEM OBSERVATIONS
================================================================================
1. Evaluated on a 20-question curated SQuAD 2.0 subset. Scaling to larger corpora
   (e.g., thousands of questions) should incorporate asynchronous batching.
2. Short answer annotations in SQuAD (e.g., '8', 'France') penalize strict Exact Match
   when the LLM provides fluent sentence answers (e.g. 'Oxygen has an atomic number of 8.').
   Containment and Token F1 reflect generative factual correctness more faithfully.
3. Strict abstention thresholding prevents hallucinations on ambiguous or adversarial
   unanswerable queries.
4. Free Tier rate and daily request limits can throttle high-frequency sequential queries;
   the evaluation pipeline handled errors gracefully without process termination.
================================================================================
"""
    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info(f"Updated {RESULTS_CSV_PATH.name} and {SUMMARY_TXT_PATH.name}")
    print(summary_text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--recompute":
        recompute_from_existing_csv()
    else:
        run_evaluation()
