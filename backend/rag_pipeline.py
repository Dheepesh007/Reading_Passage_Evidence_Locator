"""
RAG Pipeline orchestrator for Reading Passage Evidence Locator.
Connects PDF processing, text chunking, embedding generation, FAISS vector indexing,
evidence retrieval, sufficiency checking, and Gemini answer generation.
Enforces strict provenance, page citation, and abstention policies.
"""

import os
import time
import logging
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

from backend.pdf_processor import PDFProcessor
from backend.text_chunker import TextChunker
from backend.embeddings import EmbeddingManager
from backend.vector_store import FAISSVectorStore
from backend.retriever import EvidenceRetriever
from backend.llm_handler import LLMHandler, INSUFFICIENT_EVIDENCE_PHRASE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end evidence retrieval and question answering pipeline."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        gemini_model_name: str = "gemini-3.6-flash",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        similarity_threshold: float = 0.35
    ):
        """
        Initializes pipeline components.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

        # Initialize backend modules
        logger.info("Initializing RAG Pipeline modules...")
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.embedding_manager = EmbeddingManager(model_name=embedding_model_name)
        self.vector_store = FAISSVectorStore(dimension=self.embedding_manager.dimension)
        self.retriever = EvidenceRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            similarity_threshold=self.similarity_threshold
        )
        self.llm_handler = LLMHandler(model_name=gemini_model_name)

        self._indexed_files_metadata: List[Dict[str, Any]] = []
        logger.info("RAG Pipeline initialization complete.")

    def index_documents(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Ingests multiple PDF documents, extracts text page-by-page, chunks them,
        generates embeddings, and indexes them into the FAISS vector store.

        Args:
            file_paths: List of paths to PDF documents.

        Returns:
            Dict containing indexing summary statistics:
            {
                "status": "success" | "partial" | "error",
                "indexed_files": List[str],
                "failed_files": Dict[str, str],
                "total_pages": int,
                "total_chunks": int,
                "elapsed_time": float
            }
        """
        start_time = time.time()
        logger.info(f"Starting indexing for {len(file_paths)} document(s)...")

        if not file_paths:
            return {
                "status": "error",
                "message": "No file paths provided.",
                "indexed_files": [],
                "failed_files": {},
                "total_pages": 0,
                "total_chunks": 0,
                "elapsed_time": 0.0
            }

        # Step 1: Extract text from PDFs
        extraction_res = self.pdf_processor.extract_from_multiple(file_paths)
        pages = extraction_res["pages"]
        failed_files = extraction_res["failed_files"]
        processed_files = extraction_res["processed_files"]

        if not pages:
            return {
                "status": "error",
                "message": "No readable pages could be extracted from the provided files.",
                "indexed_files": processed_files,
                "failed_files": failed_files,
                "total_pages": 0,
                "total_chunks": 0,
                "elapsed_time": round(time.time() - start_time, 2)
            }

        # Step 2: Chunk pages into segments preserving document and page numbers
        chunks = self.chunker.chunk_pages(pages)
        if not chunks:
            return {
                "status": "error",
                "message": "No chunks generated from extracted text (text may be empty).",
                "indexed_files": processed_files,
                "failed_files": failed_files,
                "total_pages": len(pages),
                "total_chunks": 0,
                "elapsed_time": round(time.time() - start_time, 2)
            }

        # Step 3: Compute normalized embeddings
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_manager.get_embeddings(chunk_texts)

        # Step 4: Index into FAISS CPU vector store
        self.vector_store.add_documents(embeddings, chunks)

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"Successfully indexed {len(processed_files)} file(s), "
            f"{len(pages)} page(s), {len(chunks)} chunk(s) in {elapsed}s."
        )

        return {
            "status": "partial" if failed_files else "success",
            "indexed_files": processed_files,
            "failed_files": failed_files,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "elapsed_time": elapsed
        }

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes the full Question-Answering pipeline:
        User Question
        → Retrieve Relevant Evidence
        → Check Evidence Sufficiency
        → Send Evidence to Gemini
        → Generate Answer
        → Verify Supporting References
        → Return Answer with Sources.

        Args:
            question: User inquiry string.
            top_k: Maximum number of evidence passages to retrieve.
            threshold: Minimum similarity threshold (overrides default).

        Returns:
            Dict containing answer, sources, citations, and status:
            {
                "question": str,
                "answer": str,
                "sources": List[Dict[str, Any]],
                "abstained": bool,
                "status": "success" | "insufficient_evidence" | "error",
                "latency_seconds": float
            }
        """
        start_time = time.time()
        q_clean = question.strip() if question else ""

        if not q_clean:
            return {
                "question": question,
                "answer": "Please provide a valid question.",
                "sources": [],
                "abstained": False,
                "status": "error",
                "latency_seconds": 0.0
            }

        # Check if vector store is populated
        if self.vector_store.total_chunks == 0:
            return {
                "question": q_clean,
                "answer": "No documents are currently indexed. Please upload at least one PDF file first.",
                "sources": [],
                "abstained": False,
                "status": "error",
                "latency_seconds": 0.0
            }

        # Step 1: Retrieve relevant evidence passages
        retrieved_passages = self.retriever.retrieve(
            query=q_clean,
            top_k=top_k,
            threshold=threshold
        )

        # Step 2: Check Evidence Sufficiency
        if not self.retriever.is_evidence_sufficient(retrieved_passages, threshold=threshold):
            logger.info(f"Insufficient evidence for query: '{q_clean}'. Abstaining.")
            return {
                "question": q_clean,
                "answer": INSUFFICIENT_EVIDENCE_PHRASE,
                "sources": [],
                "abstained": True,
                "status": "insufficient_evidence",
                "latency_seconds": round(time.time() - start_time, 3)
            }

        # Step 3: Send evidence to Gemini & Generate Answer
        llm_response = self.llm_handler.generate_answer(
            query=q_clean,
            evidence_chunks=retrieved_passages
        )

        # Step 4: Verify Supporting References
        if llm_response["abstained"]:
            answer_text = INSUFFICIENT_EVIDENCE_PHRASE
            sources = []
            abstained = True
            status = "insufficient_evidence"
        elif llm_response["status"] == "error":
            answer_text = llm_response["answer"]
            sources = retrieved_passages
            abstained = False
            status = "error"
        else:
            answer_text = llm_response["answer"]
            sources = retrieved_passages
            abstained = False
            status = "success"

        elapsed = round(time.time() - start_time, 3)

        return {
            "question": q_clean,
            "answer": answer_text,
            "sources": sources,
            "abstained": abstained,
            "status": status,
            "latency_seconds": elapsed
        }

    def clear(self):
        """Clears all indexed documents and resets vector index."""
        self.vector_store.clear()
        self._indexed_files_metadata = []
        logger.info("Cleared all pipeline data.")
