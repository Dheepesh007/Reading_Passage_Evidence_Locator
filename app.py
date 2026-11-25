"""
Main application launcher for Reading Passage Evidence Locator (PS 77).
Wires the professional Gradio frontend to the RAG backend pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.ui import (
    build_ui,
    format_evidence_html,
    format_status_html,
    format_doc_list_html,
    EMPTY_EVIDENCE_HTML,
    NO_DOCS_HTML,
)

# Load environment configuration
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LAZY PIPELINE SINGLETON (avoids loading the embedding model at import time)
# ---------------------------------------------------------------------------
_pipeline = None


def get_pipeline():
    """Returns the singleton RAGPipeline, initializing it on first call."""
    global _pipeline
    if _pipeline is None:
        logger.info("First request — initializing RAG Pipeline (this loads the embedding model)...")
        from backend.rag_pipeline import RAGPipeline
        _pipeline = RAGPipeline(
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")),
        )
    return _pipeline


# ---------------------------------------------------------------------------
# CALLBACK: Process & Index uploaded PDFs
# ---------------------------------------------------------------------------

def process_documents(files):
    """
    Callback for the 'Process & Index' button.
    Receives Gradio file uploads, indexes them, and returns status + doc list HTML.
    """
    if not files:
        return (
            format_status_html("error", "No files uploaded. Please upload at least one PDF."),
            NO_DOCS_HTML,
        )

    try:
        pipeline = get_pipeline()

        # Gradio gives either NamedString paths or UploadFile objects
        file_paths = []
        for f in files:
            # Gradio File component returns filepath strings or objects with .name
            path = f if isinstance(f, str) else getattr(f, "name", str(f))
            file_paths.append(path)

        # Clear previous index and re-index with new files
        pipeline.clear()
        result = pipeline.index_documents(file_paths)

        if result["status"] == "error":
            msg = result.get("message", "Failed to process documents.")
            return (
                format_status_html("error", msg),
                NO_DOCS_HTML,
            )

        # Build success displays
        indexed = result.get("indexed_files", [])
        failed = result.get("failed_files", {})

        if failed:
            fail_names = ", ".join(Path(k).name for k in failed.keys())
            status_msg = f"Indexed {len(indexed)} file(s). Failed: {fail_names}"
            status_type = "ready"
        else:
            status_msg = f"Successfully indexed {len(indexed)} file(s) — {result['total_pages']} pages, {result['total_chunks']} chunks in {result['elapsed_time']}s"
            status_type = "ready"

        return (
            format_status_html(status_type, status_msg),
            format_doc_list_html(indexed, result),
        )

    except Exception as e:
        logger.error(f"Error processing documents: {e}", exc_info=True)
        return (
            format_status_html("error", f"Processing error: {str(e)}"),
            NO_DOCS_HTML,
        )


# ---------------------------------------------------------------------------
# CALLBACK: Ask a question
# ---------------------------------------------------------------------------

def ask_question(question: str, chat_history: list):
    """
    Callback for the Send button / Enter key.
    Takes the user question and chat history, appends the Q&A turn, and
    returns updated chat history + evidence HTML.
    """
    if not question or not question.strip():
        return chat_history, EMPTY_EVIDENCE_HTML, ""

    q_clean = question.strip()

    # Append user message immediately
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": q_clean})

    try:
        pipeline = get_pipeline()

        if pipeline.vector_store.total_chunks == 0:
            bot_msg = "⚠️ **No documents indexed.** Please upload and process PDF files before asking questions."
            chat_history.append({"role": "assistant", "content": bot_msg})
            return chat_history, EMPTY_EVIDENCE_HTML, ""

        # Execute RAG pipeline
        result = pipeline.answer_question(q_clean)
        answer = result.get("answer", "An error occurred.")
        sources = result.get("sources", [])
        status = result.get("status", "error")
        abstained = result.get("abstained", False)
        latency = result.get("latency_seconds", 0.0)

        # Build bot response
        if abstained:
            bot_msg = f"🔍 **Evidence Assessment: Insufficient**\n\n{answer}"
        elif status == "error":
            bot_msg = f"⚠️ {answer}"
        else:
            bot_msg = answer
            if latency > 0:
                bot_msg += f"\n\n---\n*⏱ Response generated in {latency}s*"

        chat_history.append({"role": "assistant", "content": bot_msg})

        # Build evidence panel
        evidence_html = format_evidence_html(sources, abstained=abstained)

        return chat_history, evidence_html, ""

    except Exception as e:
        logger.error(f"Error during question answering: {e}", exc_info=True)
        err_msg = f"⚠️ An error occurred while generating the answer: {str(e)}"
        chat_history.append({"role": "assistant", "content": err_msg})
        return chat_history, EMPTY_EVIDENCE_HTML, ""


# ---------------------------------------------------------------------------
# CALLBACK: Clear All
# ---------------------------------------------------------------------------

def clear_all():
    """Resets the pipeline, chat, evidence, and status displays."""
    global _pipeline
    if _pipeline is not None:
        _pipeline.clear()

    return (
        [],  # chat history
        EMPTY_EVIDENCE_HTML,
        format_status_html("idle", "Waiting for document upload..."),
        NO_DOCS_HTML,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    demo = build_ui(
        process_callback=process_documents,
        ask_callback=ask_question,
        clear_callback=clear_all,
    )
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=getattr(demo, "_rpel_theme", None),
        css=getattr(demo, "_rpel_css", None),
    )


if __name__ == "__main__":
    main()
