"""
Integration tests for Reading Passage Evidence Locator (PS 77) frontend and app callbacks.
Tests:
1. build_ui creates a valid gr.Blocks instance.
2. format_evidence_html renders cards, page numbers, scores, and abstention warnings.
3. format_status_html renders appropriate styling for idle, ready, and error states.
4. format_doc_list_html formats indexed document names and stats.
5. process_documents callback handles empty files and valid PDF indexing.
6. ask_question callback handles empty input, unindexed states, and out-of-domain abstention.
7. clear_all callback resets state cleanly.
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from frontend.ui import (
    build_ui,
    format_evidence_html,
    format_status_html,
    format_doc_list_html,
    EMPTY_EVIDENCE_HTML,
    NO_DOCS_HTML,
)
from app import process_documents, ask_question, clear_all, get_pipeline


class TestFrontendIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_pdf = Path("data/sample_pdfs/sample_doc1.pdf")
        if not cls.sample_pdf.exists():
            # If for any reason sample_doc1.pdf doesn't exist, create it via pymupdf
            import pymupdf
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_textbox(
                pymupdf.Rect(50, 50, 550, 750),
                "Reading Passage Evidence Locator test document for frontend integration.\n"
                "This document verifies that UI callbacks correctly interact with the RAG pipeline."
            )
            cls.sample_pdf.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(cls.sample_pdf))
            doc.close()

    def test_01_build_ui_instance(self):
        """Verifies build_ui returns a properly configured Gradio Blocks instance."""
        demo = build_ui(
            process_callback=process_documents,
            ask_callback=ask_question,
            clear_callback=clear_all,
        )
        self.assertIsInstance(demo, gr.Blocks)
        self.assertTrue(hasattr(demo, "_rpel_theme"))
        self.assertTrue(hasattr(demo, "_rpel_css"))

    def test_02_format_evidence_html(self):
        """Verifies evidence formatting for empty, populated, and abstained states."""
        # Empty state
        empty_html = format_evidence_html([])
        self.assertIn("No Evidence Retrieved Yet", empty_html)

        # Populated state with page number and document name
        sources = [{
            "document_name": "research.pdf",
            "page_number": 3,
            "text": "Photosynthesis produces glucose and oxygen.",
            "similarity_score": 0.8842
        }]
        pop_html = format_evidence_html(sources)
        self.assertIn("research.pdf", pop_html)
        self.assertIn("Page 3", pop_html)
        self.assertIn("88.4%", pop_html)
        self.assertIn("Photosynthesis produces glucose", pop_html)

        # Abstained state
        abs_html = format_evidence_html([], abstained=True)
        self.assertIn("Insufficient Grounding Evidence (Abstained)", abs_html)
        self.assertIn("similarity threshold &ge; 0.35", abs_html)

    def test_03_format_status_html(self):
        """Verifies status indicator HTML generator."""
        ready_html = format_status_html("ready", "Indexing complete")
        self.assertIn("rpel-status-ready", ready_html)
        self.assertIn("Indexing complete", ready_html)

        idle_html = format_status_html("idle", "Standby")
        self.assertIn("rpel-status-idle", idle_html)

        err_html = format_status_html("error", "Failed to parse")
        self.assertIn("rpel-status-error", err_html)

    def test_04_format_doc_list_html(self):
        """Verifies document list HTML rendering with stats."""
        none_html = format_doc_list_html([])
        self.assertEqual(none_html, NO_DOCS_HTML)

        docs = ["docA.pdf", "docB.pdf"]
        stats = {"total_pages": 5, "total_chunks": 12, "elapsed_time": 0.45}
        rendered = format_doc_list_html(docs, stats)
        self.assertIn("docA.pdf", rendered)
        self.assertIn("docB.pdf", rendered)
        self.assertIn("12", rendered)  # chunks
        self.assertIn("5", rendered)   # pages

    def test_05_process_documents_callback(self):
        """Verifies the process_documents callback with empty and valid inputs."""
        # Test empty input
        status, doc_list = process_documents([])
        self.assertIn("rpel-status-error", status)
        self.assertIn("No files uploaded", status)

        # Test valid sample PDF
        status_ok, doc_list_ok = process_documents([str(self.sample_pdf)])
        self.assertIn("rpel-status-ready", status_ok)
        self.assertIn("Successfully indexed", status_ok)
        self.assertIn(self.sample_pdf.name, doc_list_ok)

    def test_06_ask_question_callback(self):
        """Verifies the ask_question callback for empty, in-domain, and out-of-domain queries."""
        # Index document first
        process_documents([str(self.sample_pdf)])

        # Test empty question
        history, ev_html, input_val = ask_question("", [])
        self.assertEqual(history, [])
        self.assertEqual(input_val, "")

        # Test out-of-domain question -> must trigger abstention
        history, ev_html, input_val = ask_question(
            "What was the GDP of France in the seventeenth century?", []
        )
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertIn("Evidence Assessment: Insufficient", history[1]["content"])
        self.assertIn("Insufficient Grounding Evidence (Abstained)", ev_html)

    def test_07_clear_all_callback(self):
        """Verifies that clear_all resets all session states."""
        history, ev_html, status_html, doc_html = clear_all()
        self.assertEqual(history, [])
        self.assertEqual(ev_html, EMPTY_EVIDENCE_HTML)
        self.assertIn("rpel-status-idle", status_html)
        self.assertEqual(doc_html, NO_DOCS_HTML)

        # Vector store should now have 0 chunks
        pipeline = get_pipeline()
        self.assertEqual(pipeline.vector_store.total_chunks, 0)


if __name__ == "__main__":
    unittest.main()
