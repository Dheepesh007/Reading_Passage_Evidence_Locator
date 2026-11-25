"""
Prepare SQuAD 2.0 Evaluation Dataset for Reading Passage Evidence Locator (PS 77).

Extracts a balanced evaluation set (10 answerable, 10 unanswerable questions)
from SQuAD 2.0 dev-v2.0.json, generates clean reading passage PDFs, and produces
the ground-truth evaluation CSV.
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Use pymupdf directly
try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        sys.exit("Error: PyMuPDF is required. Please install via: pip install PyMuPDF")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_squad")

# Relative project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_JSON_PATH = PROJECT_ROOT / "data" / "squad_dataset" / "dev-v2.0.json"
FALLBACK_JSON_PATH = PROJECT_ROOT / "data" / "sample_pdfs" / "squad_dataset" / "dev-v2.0.json"
OUTPUT_PDF_DIR = PROJECT_ROOT / "data" / "sample_pdfs" / "squad"
OUTPUT_CSV_PATH = PROJECT_ROOT / "evaluation" / "squad_test_questions.csv"


def resolve_json_path() -> Path:
    """Locates dev-v2.0.json across primary and fallback locations."""
    if PRIMARY_JSON_PATH.exists():
        return PRIMARY_JSON_PATH
    if FALLBACK_JSON_PATH.exists():
        logger.info(f"Primary path not found, using fallback: {FALLBACK_JSON_PATH}")
        return FALLBACK_JSON_PATH
    raise FileNotFoundError(
        f"dev-v2.0.json not found at '{PRIMARY_JSON_PATH}' or '{FALLBACK_JSON_PATH}'. "
        "Please ensure the SQuAD 2.0 dataset is downloaded."
    )


def load_squad_data(json_path: Path) -> List[Dict[str, Any]]:
    """Loads and validates SQuAD 2.0 JSON format."""
    logger.info(f"Loading SQuAD 2.0 dataset from: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file format in '{json_path}': {e}")
    except Exception as e:
        raise RuntimeError(f"Could not read '{json_path}': {e}")

    if "data" not in payload or not isinstance(payload["data"], list):
        raise ValueError(f"Unexpected SQuAD structure: 'data' list missing in '{json_path}'.")

    return payload["data"]


def select_balanced_samples(
    articles: List[Dict[str, Any]],
    num_passages: int = 10
) -> List[Dict[str, Any]]:
    """
    Selects balanced passages across diverse articles.
    Each passage contains:
    - 1 Answerable question (is_impossible == False) with ground-truth answer.
    - 1 Unanswerable question (is_impossible == True) with empty expected answer.
    """
    selected_passages: List[Dict[str, Any]] = []

    for article in articles:
        title = article.get("title", "Unknown_Article")
        paragraphs = article.get("paragraphs", [])

        # Find a paragraph in this article that has both answerable and unanswerable questions
        for p_idx, para in enumerate(paragraphs):
            context = para.get("context", "").strip()
            qas = para.get("qas", [])

            if not context or len(context) < 150:
                continue

            ans_candidates = []
            unans_candidates = []

            for q in qas:
                is_impossible = q.get("is_impossible", False)
                q_text = q.get("question", "").strip()
                answers = q.get("answers", [])

                if not q_text:
                    continue

                if not is_impossible and answers:
                    # Valid answerable question
                    first_ans = answers[0].get("text", "").strip()
                    if first_ans:
                        ans_candidates.append({
                            "question_id": q.get("id"),
                            "question": q_text,
                            "is_answerable": True,
                            "expected_answer": first_ans
                        })
                elif is_impossible:
                    # Valid unanswerable question
                    unans_candidates.append({
                        "question_id": q.get("id"),
                        "question": q_text,
                        "is_answerable": False,
                        "expected_answer": ""
                    })

            # Check if this paragraph yields at least 1 of each
            if ans_candidates and unans_candidates:
                passage_num = len(selected_passages) + 1
                passage_id = f"squad_p{passage_num:02d}"

                selected_passages.append({
                    "passage_id": passage_id,
                    "source_article_title": title,
                    "paragraph_index": p_idx,
                    "context": context,
                    "answerable_qa": ans_candidates[0],
                    "unanswerable_qa": unans_candidates[0]
                })
                break  # Move to next article for diversity

        if len(selected_passages) >= num_passages:
            break

    if len(selected_passages) < num_passages:
        raise ValueError(
            f"Could only find {len(selected_passages)} balanced passages; required {num_passages}."
        )

    return selected_passages


def create_passage_pdf(
    passage: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Creates a clean, human-readable reading passage PDF using PyMuPDF.
    Contains ONLY the passage text, article title, and passage ID.
    Questions and answers are NOT included.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    passage_id = passage["passage_id"]
    clean_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in passage["source_article_title"])
    filename = f"{passage_id}_{clean_title}.pdf"
    pdf_path = output_dir / filename

    doc = pymupdf.open()
    # Standard A4: 595 x 842 points
    page = doc.new_page(width=595, height=842)

    # Margins
    left_margin = 54
    top_margin = 54
    right_margin = 541
    bottom_margin = 788

    # Header block: Article Title & Passage ID
    title_text = f"Article: {passage['source_article_title']}\nPassage Reference: {passage_id}"
    header_rect = pymupdf.Rect(left_margin, top_margin, right_margin, top_margin + 50)
    page.insert_textbox(
        header_rect,
        title_text,
        fontsize=12,
        color=(0.12, 0.23, 0.37)  # Navy accent (#1e3a5f)
    )

    # Decorative separator line
    line_y = top_margin + 55
    page.draw_line(
        pymupdf.Point(left_margin, line_y),
        pymupdf.Point(right_margin, line_y),
        color=(0.31, 0.27, 0.90),  # Indigo accent (#4f46e5)
        width=1.5
    )

    # Passage Content (Only reading passage text)
    body_top = line_y + 20
    body_rect = pymupdf.Rect(left_margin, body_top, right_margin, bottom_margin - 30)
    
    # Insert readable context text
    context_text = passage["context"]
    # Handle text overflow onto a second page if context is very long
    remaining_text = page.insert_textbox(
        body_rect,
        context_text,
        fontsize=11,
        color=(0.1, 0.1, 0.1)
    )

    # If text overflowed the rectangle, add subsequent pages
    while remaining_text and len(context_text) > 0 and remaining_text < 0:
        page2 = doc.new_page(width=595, height=842)
        body_rect2 = pymupdf.Rect(left_margin, top_margin, right_margin, bottom_margin - 30)
        remaining_text = page2.insert_textbox(
            body_rect2,
            context_text,
            fontsize=11,
            color=(0.1, 0.1, 0.1)
        )
        break

    # Footer note
    footer_text = "Reading Passage Evidence Locator · PS 77 Evaluation Dataset"
    footer_rect = pymupdf.Rect(left_margin, bottom_margin - 15, right_margin, bottom_margin)
    page.insert_textbox(
        footer_rect,
        footer_text,
        fontsize=8,
        color=(0.58, 0.64, 0.72)
    )

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


def export_evaluation_csv(
    passages: List[Dict[str, Any]],
    output_csv_path: Path
) -> List[Dict[str, Any]]:
    """
    Exports exactly 20 questions (10 answerable, 10 unanswerable) to CSV.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    for p in passages:
        p_id = p["passage_id"]
        title = p["source_article_title"]

        # 1. Answerable Question
        ans_qa = p["answerable_qa"]
        all_rows.append({
            "question_id": ans_qa["question_id"],
            "passage_id": p_id,
            "question": ans_qa["question"],
            "is_answerable": True,
            "expected_answer": ans_qa["expected_answer"],
            "source_article_title": title
        })

        # 2. Unanswerable Question
        unans_qa = p["unanswerable_qa"]
        all_rows.append({
            "question_id": unans_qa["question_id"],
            "passage_id": p_id,
            "question": unans_qa["question"],
            "is_answerable": False,
            "expected_answer": "",
            "source_article_title": title
        })

    fieldnames = [
        "question_id",
        "passage_id",
        "question",
        "is_answerable",
        "expected_answer",
        "source_article_title"
    ]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return all_rows


def main():
    logger.info("=== SQuAD 2.0 Evaluation Preparation Pipeline ===")

    # 1. Resolve and Load SQuAD JSON
    try:
        json_path = resolve_json_path()
        articles = load_squad_data(json_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 2. Select Balanced Samples (10 passages -> 10 answerable, 10 unanswerable)
    try:
        passages = select_balanced_samples(articles, num_passages=10)
    except Exception as e:
        logger.error(f"Sample selection error: {e}")
        sys.exit(1)

    # 3. Generate Passage PDFs
    generated_pdfs: List[Path] = []
    for p in passages:
        try:
            pdf_path = create_passage_pdf(p, OUTPUT_PDF_DIR)
            generated_pdfs.append(pdf_path)
            logger.info(f"Generated PDF: {pdf_path.name}")
        except Exception as e:
            logger.error(f"Error creating PDF for {p['passage_id']}: {e}")
            sys.exit(1)

    # 4. Export Evaluation CSV
    try:
        exported_rows = export_evaluation_csv(passages, OUTPUT_CSV_PATH)
        logger.info(f"Exported evaluation CSV: {OUTPUT_CSV_PATH}")
    except Exception as e:
        logger.error(f"Error writing evaluation CSV: {e}")
        sys.exit(1)

    # 5. Print Execution Summary
    num_answerable = sum(1 for r in exported_rows if r["is_answerable"] is True)
    num_unanswerable = sum(1 for r in exported_rows if r["is_answerable"] is False)
    unique_passages = len(set(r["passage_id"] for r in exported_rows))

    print("\n" + "=" * 60)
    print("SQuAD 2.0 Evaluation Dataset Preparation Summary")
    print("=" * 60)
    print(f"Number of answerable questions   : {num_answerable}")
    print(f"Number of unanswerable questions : {num_unanswerable}")
    print(f"Number of unique passages        : {unique_passages}")
    print(f"Number of PDFs generated         : {len(generated_pdfs)}")
    print(f"Total questions exported         : {len(exported_rows)}")
    print("-" * 60)
    print(f"Output PDF directory             : {OUTPUT_PDF_DIR}")
    print(f"Output Evaluation CSV            : {OUTPUT_CSV_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
