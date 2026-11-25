# Methodology & Technical Approach

## 1. Document Ingestion & Page-Preserving Text Processing
Standard PDF text parsers often discard pagination or merge headers/footers haphazardly. In our methodology:
- **PyMuPDF (`fitz`)** accesses page objects directly using `doc.load_page(page_number)`.
- Extracted text maintains explicit mapping: `(doc_id, page_number, raw_text)`.
- Whitespace normalization and ligature cleanups are performed without disrupting word boundaries.

## 2. Chunking Strategy
- **Chunk Size**: ~400–500 characters / tokens with a 100-character overlap.
- **Boundary Preservation**: Avoid splitting sentences mid-clause.
- **Traceability Guarantee**: Each chunk dictionary holds:
  ```python
  {
      "chunk_id": 42,
      "document_name": "research_paper.pdf",
      "page_number": 3,
      "text": "..."
  }
  ```

## 3. Embedding & Indexing
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors). Fast, accurate, and optimized for CPU inference.
- **Vector Index**: FAISS `IndexFlatIP`. All embeddings are L2-normalized upon creation, converting Inner Product queries directly into exact Cosine Similarity calculations.
- **Caching**: The model weights remain loaded in a persistent singleton in memory across user queries.

## 4. Evidence Retrieval & Thresholding
- For any user question $Q$:
  1. Generate query embedding $E_Q = \text{Embed}(Q)$.
  2. Query FAISS index for top-$k$ nearest chunks ($k=5$).
  3. Inspect cosine similarity scores $S_i$.
  4. If $\max(S_i) < \tau$ (where threshold $\tau \approx 0.35$), trigger early abstention before calling the LLM.

## 5. Evidence-Grounded LLM Generation & Abstention
- Filtered chunks are formatted into a structured context block:
  ```
  [Source 1] (Document: filename.pdf, Page: 4):
  "..."
  
  [Source 2] (Document: filename.pdf, Page: 5):
  "..."
  ```
- **Prompt Guardrails**:
  - Direct the Gemini model to answer using **only** facts present in the source chunks.
  - Require sentence-level citations in the format `[Doc: filename, Page: X]`.
  - If the context does not explicitly provide the answer, output the exact phrase:  
    `"INSUFFICIENT_EVIDENCE: The uploaded document(s) do not contain sufficient evidence to answer this question."`

## 6. Evaluation Protocol
- Benchmark test set in `evaluation/test_questions.csv` containing:
  - Factual questions with known ground-truth pages.
  - Out-of-domain questions to verify abstention rate.
  - Paraphrased and multi-hop queries.
- Measure retrieval accuracy (Hit Rate@k), citation precision, and hallucination rate.
