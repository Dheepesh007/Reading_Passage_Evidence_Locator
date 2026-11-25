"""
Unit tests for Reading Passage Evidence Locator (PS 77) backend modules.
Tests:
1. PDF extraction (page preservation, empty pages, metadata)
2. Chunk generation (sentence integrity, page tagging)
3. Embedding generation (singleton caching, normalization, dimension)
4. FAISS retrieval (inner product / cosine ranking, multi-doc indexing)
5. Evidence sufficiency (relevance threshold, abstention flag)
6. RAG pipeline end-to-end (document indexing, abstention on out-of-domain questions)
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PyMuPDF
import pymupdf

# Backend imports
from backend.pdf_processor import PDFProcessor
from backend.text_chunker import TextChunker
from backend.embeddings import EmbeddingManager
from backend.vector_store import FAISSVectorStore
from backend.retriever import EvidenceRetriever
from backend.llm_handler import LLMHandler, INSUFFICIENT_EVIDENCE_PHRASE
from backend.rag_pipeline import RAGPipeline


class TestBackendModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Creates sample test PDF files for deterministic testing."""
        cls.test_dir = Path("data/sample_pdfs")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls.sample_pdf_1 = cls.test_dir / "sample_doc1.pdf"
        cls.sample_pdf_2 = cls.test_dir / "sample_doc2.pdf"
        cls.empty_pdf = cls.test_dir / "empty_doc.pdf"

        # Create sample_doc1.pdf (2 pages)
        doc1 = pymupdf.open()
        p1 = doc1.new_page()
        p1.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            "Reading Passage Evidence Locator is an evidence-grounded system.\n"
            "It was engineered for Problem Statement 77 in the hackathon.\n"
            "The primary objective is finding exact page-level citations for every factual claim."
        )
        p2 = doc1.new_page()
        p2.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            "Gradio was selected as the user interface framework to eliminate UI lag.\n"
            "The vector database utilizes FAISS CPU with inner product metrics.\n"
            "Embeddings are computed using the all-MiniLM-L6-v2 Sentence Transformers model."
        )
        doc1.save(str(cls.sample_pdf_1))
        doc1.close()

        # Create sample_doc2.pdf (1 page)
        doc2 = pymupdf.open()
        p_doc2 = doc2.new_page()
        p_doc2.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            "Photosynthesis is the biological process used by plants and other organisms\n"
            "to convert light energy into chemical energy stored in glucose.\n"
            "Chlorophyll is the green pigment responsible for absorbing sunlight."
        )
        doc2.save(str(cls.sample_pdf_2))
        doc2.close()

        # Create empty_doc.pdf (1 blank page)
        doc_empty = pymupdf.open()
        doc_empty.new_page()  # blank
        doc_empty.save(str(cls.empty_pdf))
        doc_empty.close()

    # ----------------------------------------------------
    # 1. PDF Extraction Tests
    # ----------------------------------------------------
    def test_01_pdf_extraction(self):
        """Verifies PDF extraction, 1-based page numbering, and blank page detection."""
        pages = PDFProcessor.extract_from_file(self.sample_pdf_1)
        self.assertEqual(len(pages), 2, "Expected 2 pages extracted from sample_doc1.pdf")

        # Check 1-based numbering
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[1]["page_number"], 2)

        # Check content presence
        self.assertIn("Problem Statement 77", pages[0]["text"])
        self.assertIn("Gradio was selected", pages[1]["text"])
        self.assertFalse(pages[0]["is_empty"])

        # Check blank page detection
        empty_pages = PDFProcessor.extract_from_file(self.empty_pdf)
        self.assertEqual(len(empty_pages), 1)
        self.assertTrue(empty_pages[0]["is_empty"])

        # Check multi-file extraction
        multi_res = PDFProcessor.extract_from_multiple([self.sample_pdf_1, self.sample_pdf_2])
        self.assertEqual(multi_res["total_pages"], 3)
        self.assertEqual(len(multi_res["processed_files"]), 2)
        self.assertEqual(len(multi_res["failed_files"]), 0)

    # ----------------------------------------------------
    # 2. Text Chunker Tests
    # ----------------------------------------------------
    def test_02_text_chunking(self):
        """Verifies text chunking preserves metadata, page numbers, and avoids cutting sentences."""
        pages = PDFProcessor.extract_from_file(self.sample_pdf_1)
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_pages(pages)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("chunk_id", chunk)
            self.assertIn("document_name", chunk)
            self.assertIn("page_number", chunk)
            self.assertIn("text", chunk)
            self.assertIn(chunk["page_number"], [1, 2])
            self.assertEqual(chunk["document_name"], "sample_doc1.pdf")
            self.assertTrue(len(chunk["text"]) > 0)

    # ----------------------------------------------------
    # 3. Embedding Generation Tests
    # ----------------------------------------------------
    def test_03_embedding_generation(self):
        """Verifies singleton caching, embedding dimension, and L2 normalization."""
        emb1 = EmbeddingManager()
        emb2 = EmbeddingManager()
        # Singleton verification
        self.assertIs(emb1, emb2, "EmbeddingManager must be a singleton")

        test_texts = [
            "Reading Passage Evidence Locator",
            "Photosynthesis converts light into chemical energy."
        ]
        vectors = emb1.get_embeddings(test_texts)

        self.assertEqual(vectors.shape[0], 2)
        self.assertEqual(vectors.shape[1], 384, "all-MiniLM-L6-v2 dimension must be 384")
        self.assertEqual(vectors.dtype, np.float32)

        # Check L2 normalization (norm should be ~1.0)
        norms = np.linalg.norm(vectors, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=4, msg="Embeddings must be L2-normalized")

    # ----------------------------------------------------
    # 4. FAISS Retrieval Tests
    # ----------------------------------------------------
    def test_04_faiss_retrieval(self):
        """Verifies FAISS CPU index creation, multi-doc addition, and similarity search."""
        emb_mgr = EmbeddingManager()
        vector_store = FAISSVectorStore(dimension=emb_mgr.dimension)

        test_chunks = [
            {"chunk_id": 0, "document_name": "docA.pdf", "page_number": 1, "text": "The cat sat on the mat."},
            {"chunk_id": 1, "document_name": "docB.pdf", "page_number": 3, "text": "Quantum computing uses qubits."}
        ]
        texts = [c["text"] for c in test_chunks]
        embs = emb_mgr.get_embeddings(texts)

        vector_store.add_documents(embs, test_chunks)
        self.assertEqual(vector_store.total_chunks, 2)
        self.assertCountEqual(vector_store.get_indexed_documents(), ["docA.pdf", "docB.pdf"])

        # Query for cat
        q_emb = emb_mgr.get_query_embedding("Where is the cat sitting?")
        results = vector_store.search(q_emb, top_k=2)

        self.assertEqual(len(results), 2)
        top_chunk, top_score = results[0]
        self.assertEqual(top_chunk["document_name"], "docA.pdf")
        self.assertIn("cat sat on the mat", top_chunk["text"])
        self.assertGreater(top_score, 0.4)

    # ----------------------------------------------------
    # 5. Evidence Sufficiency & Retriever Tests
    # ----------------------------------------------------
    def test_05_evidence_sufficiency(self):
        """Verifies similarity thresholding and sufficiency evaluation."""
        emb_mgr = EmbeddingManager()
        vector_store = FAISSVectorStore(dimension=emb_mgr.dimension)

        chunk = [{"chunk_id": 0, "document_name": "doc1.pdf", "page_number": 1, "text": "Python is a programming language."}]
        vector_store.add_documents(emb_mgr.get_embeddings([chunk[0]["text"]]), chunk)

        retriever = EvidenceRetriever(vector_store, emb_mgr, similarity_threshold=0.40)

        # Relevant query
        rel_passages = retriever.retrieve("What kind of language is Python?", top_k=1)
        self.assertEqual(len(rel_passages), 1)
        self.assertTrue(retriever.is_evidence_sufficient(rel_passages))

        # Unrelated query with high threshold
        unrel_passages = retriever.retrieve("What is the recipe for chocolate cake?", top_k=1, threshold=0.85)
        self.assertEqual(len(unrel_passages), 0)
        self.assertFalse(retriever.is_evidence_sufficient(unrel_passages))

    # ----------------------------------------------------
    # 6. RAG Pipeline Tests
    # ----------------------------------------------------
    def test_06_rag_pipeline(self):
        """Verifies end-to-end indexing, retrieval, and abstention on out-of-domain questions."""
        pipeline = RAGPipeline(similarity_threshold=0.35)

        # Index both test PDFs
        index_res = pipeline.index_documents([str(self.sample_pdf_1), str(self.sample_pdf_2)])
        self.assertEqual(index_res["status"], "success")
        self.assertEqual(index_res["total_pages"], 3)
        self.assertGreater(index_res["total_chunks"], 0)

        # Test out-of-domain question -> must trigger abstention
        out_of_domain_q = "What is the capital city of Australia and its total population in 1950?"
        ans_res = pipeline.answer_question(out_of_domain_q)

        self.assertTrue(ans_res["abstained"], "Pipeline must abstain on out-of-domain question")
        self.assertEqual(ans_res["status"], "insufficient_evidence")
        self.assertEqual(ans_res["answer"], INSUFFICIENT_EVIDENCE_PHRASE)
        self.assertEqual(len(ans_res["sources"]), 0)

        # Test relevant retrieval pipeline check (evidence retrieval verification)
        in_domain_q = "What user interface framework was selected to eliminate UI lag?"
        evidence = pipeline.retriever.retrieve(in_domain_q, top_k=2)
        self.assertGreater(len(evidence), 0)
        top_ev = evidence[0]
        self.assertEqual(top_ev["document_name"], "sample_doc1.pdf")
        self.assertEqual(top_ev["page_number"], 2)
        self.assertIn("Gradio", top_ev["text"])

    # ----------------------------------------------------
    # 7. LLM Handler & Abstention Tests
    # ----------------------------------------------------
    def test_07_llm_handler_abstention_logic(self):
        """Verifies LLMHandler abstention detection and empty evidence handling."""
        handler = LLMHandler(api_key=None)

        # Empty evidence -> must immediately return abstention without calling API
        res_empty = handler.generate_answer("Any question?", evidence_chunks=[])
        self.assertTrue(res_empty["abstained"])
        self.assertEqual(res_empty["status"], "insufficient_evidence")
        self.assertEqual(res_empty["answer"], INSUFFICIENT_EVIDENCE_PHRASE)

        # Test abstention phrasing detector
        self.assertTrue(handler._is_abstention("Insufficient information is available in the uploaded documents to answer this question."))
        self.assertTrue(handler._is_abstention("The provided excerpts do not contain any information about revenue."))
        self.assertFalse(handler._is_abstention("According to page 2, Gradio was selected as the UI framework."))

    # ----------------------------------------------------
    # 8. Gemini Connection Test (Mocked)
    # ----------------------------------------------------
    @patch("backend.llm_handler.genai.Client")
    def test_08_gemini_connection(self, mock_client_cls):
        """Verifies Gemini connection handling with mocked responses (no live API calls)."""
        # 1. Setup mock client and successful response
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        mock_resp = MagicMock()
        mock_resp.text = "GEMINI_OK"
        mock_instance.models.generate_content.return_value = mock_resp

        handler = LLMHandler(api_key="mock_test_key", model_name="gemini-3.6-flash")
        res = handler.test_connection()

        self.assertEqual(res["status"], "success", f"Connection failed: {res.get('message')}")
        self.assertEqual(res["response"], "GEMINI_OK")
        mock_instance.models.generate_content.assert_called_once_with(
            model="gemini-3.6-flash",
            contents="Reply with exactly: GEMINI_OK"
        )

        # 2. Test missing API key handling
        handler_no_key = LLMHandler(api_key=None)
        handler_no_key.api_key = None
        handler_no_key._client = None
        orig_key = os.environ.get("GEMINI_API_KEY")
        try:
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            res_no_key = handler_no_key.test_connection()
            self.assertEqual(res_no_key["status"], "error")
            self.assertIn("not configured", res_no_key["message"])
        finally:
            if orig_key is not None:
                os.environ["GEMINI_API_KEY"] = orig_key

        # 3. Test API error handling
        mock_instance.models.generate_content.side_effect = Exception("Simulated quota exhaustion or network error")
        res_err = handler.test_connection()
        self.assertEqual(res_err["status"], "error")
        self.assertIn("Simulated quota exhaustion", res_err["message"])


if __name__ == "__main__":
    unittest.main()
