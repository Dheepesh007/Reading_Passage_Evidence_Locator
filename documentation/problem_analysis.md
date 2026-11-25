# Problem Analysis: PS 77 – Reading Passage Evidence Locator

## 1. Problem Statement
In academic research, legal discovery, standard reading comprehension, and study workflows, users spend an excessive amount of time manually skimming dense, multi-page PDFs to find specific pieces of factual evidence that substantiate answers to their questions.

Conventional search tools (such as simple Ctrl+F keywords) fail when queries are phrased conceptually or use synonyms. Conversely, standard Large Language Model (LLM) Q&A tools frequently hallucinate facts or cite non-existent page numbers, leading to misinformation and lack of auditability.

## 2. Core Objectives
1. **Accurate Evidence Retrieval**: Find the exact passages in multi-document PDFs that directly answer the user's inquiry.
2. **Deterministic Provenance**: Preserve and display the exact page numbers, document names, and snippet contexts for every generated statement.
3. **Strict Abstention Policy**: When the provided texts do not contain sufficient evidence to answer a question, explicitly abstain from answering instead of guessing or fabricating facts.
4. **Fast, Lightweight Local Execution**: Run smoothly on standard CPU environments using Gradio and FAISS without UI lag or bloated dependencies.

## 3. Key Technical Challenges & Mitigations

| Challenge | Mitigation |
| :--- | :--- |
| **Document Formatting & Multi-column Layouts** | Use PyMuPDF (fitz) with structured block and page-level extraction to capture clean text order while maintaining page indices. |
| **Context Fragmentation across Chunks** | Implement overlapping sentence/token-aware text chunking that retains parent page numbers and section offsets. |
| **Model Hallucinations & Fabricated Citations** | Enforce strict system prompting with Gemini API: require the model to cite exact supporting quotes and page numbers, and abstain if confidence or coverage is low. |
| **Latency and UI Responsiveness** | Migrate from Streamlit to Gradio to eliminate reactive reruns and UI lag. Cache the Sentence Transformer embedding model in memory to prevent reload overhead. |
| **Memory / CPU Constraints** | Use `faiss-cpu` with lightweight embeddings (`all-MiniLM-L6-v2`) to ensure near-instant vector indexing on CPU. |
