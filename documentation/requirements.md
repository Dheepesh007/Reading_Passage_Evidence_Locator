# Requirements Specification

## 1. Functional Requirements (FR)

- **FR-1: Multi-PDF Upload**: The system must allow users to upload one or multiple PDF documents simultaneously.
- **FR-2: Text & Metadata Extraction**: The system must extract clean text while retaining document names and original 1-based page numbers.
- **FR-3: Intelligent Text Chunking**: Documents must be segmented into chunks (e.g. 300–500 tokens) with configurable overlap (50–100 tokens), preserving page-level traceability.
- **FR-4: In-Memory / Local Vector Indexing**: The system must compute dense vector representations via Sentence Transformers and index them using FAISS CPU.
- **FR-5: Semantic Passage Retrieval**: For any user query, the retriever must rank and fetch the top-k most relevant text chunks based on cosine similarity.
- **FR-6: Evidence-Grounded Answer Generation**: Gemini API must synthesize an answer based *only* on retrieved passages.
- **FR-7: Sentence-Level Citations**: Answers must cite supporting sentences and link them explicitly to their document and page number.
- **FR-8: Abstention Mechanism**: If retrieved chunks fall below a similarity relevance threshold or do not contain verifiable facts answering the query, the system must return an explicit abstention response (e.g., *"The uploaded documents do not contain sufficient evidence to answer this question."*).
- **FR-9: Error Handling**: Invalid PDFs, password-protected files, empty uploads, and API rate-limit errors must trigger clear user-facing alerts.

## 2. Non-Functional Requirements (NFR)

- **NFR-1: CPU Friendliness**: Vector indexing and inference must run comfortably on standard CPUs without GPU dependencies.
- **NFR-2: Performance & Latency**: UI interaction and vector retrieval should complete within sub-second times; total RAG turn time under 3–5 seconds on broadband connections.
- **NFR-3: UI Responsiveness**: Gradio Blocks interface must avoid unnecessary full-page reruns and UI freezing.
- **NFR-4: Security**: Gemini API keys must be loaded from `.env` and never logged or exposed in client responses.
- **NFR-5: Modularity**: Clean separation of concerns between PDF processing, chunking, embedding, retrieval, LLM interaction, and Gradio UI.
