# 📖 Reading Passage Evidence Locator

**Problem Statement:** PS 77 – Reading Passage Evidence Locator  
**Technology Stack:** Python 3.12, Gradio, Google Gemini API (`google-genai`), Sentence Transformers, FAISS CPU, PyMuPDF, python-dotenv

---

## 🎯 Overview

**Reading Passage Evidence Locator (RPEL)** is an evidence-grounded Retrieval-Augmented Generation (RAG) platform designed to eliminate hallucinations in document analysis. When users upload multi-page PDF documents and ask questions, the system extracts the exact factual passages, cites source document names and 1-based page numbers, and strictly **abstains** when sufficient evidence is absent.

### Key Architectural Pillars:
- **Gradio Blocks Interface**: Academic AI SaaS-inspired UI featuring an elegant light theme with navy/indigo accents, dark mode support, drag-and-drop PDF ingestion, live indexing statistics, conversational Q&A, and an expandable evidence panel.
- **Strict Evidence Grounding**: Answers are synthesized exclusively from retrieved document chunks. The prompt forces zero-shot factual citation with `[Document: <name>, Page: <number>]`.
- **Guaranteed Abstention Policy**: If no retrieved passages meet the semantic threshold ($\ge 0.35$), the system immediately returns:  
  `"Insufficient information is available in the uploaded documents to answer this question."`
- **CPU-Optimized & Scalable**: Employs `faiss.IndexFlatIP` on normalized vectors and loads the `all-MiniLM-L6-v2` embedding model only once via a thread-safe singleton pattern.
- **Modern SDK**: Fully integrated with the official `google-genai` SDK (`from google import genai`).

---

## 📂 Project Structure

```
Reading_Passage_Evidence_Locator/
│
├── app.py                     # Main application entry point & Gradio launcher
├── requirements.txt           # Python package dependencies (compatible with Python 3.12)
├── .env.example               # Environment variables template
├── .gitignore                 # Version control exclusions
├── README.md                  # Comprehensive documentation and run guide
│
├── backend/                   # Core RAG engine modules
│   ├── __init__.py
│   ├── pdf_processor.py       # Page-by-page text extraction with PyMuPDF
│   ├── text_chunker.py        # Sentence-preserving chunking with page metadata
│   ├── embeddings.py          # Singleton Sentence Transformers wrapper (all-MiniLM-L6-v2)
│   ├── vector_store.py        # FAISS CPU vector index with cosine similarity
│   ├── retriever.py           # Top-K passage ranking & sufficiency evaluation
│   ├── llm_handler.py         # Google Gemini integration via google-genai
│   └── rag_pipeline.py        # End-to-end RAG orchestrator
│
├── frontend/                  # User Interface
│   ├── __init__.py
│   └── ui.py                  # Professional Gradio Blocks layout, custom CSS, & components
│
├── tests/                     # Automated Test Suites
│   ├── __init__.py
│   ├── test_backend.py        # Unit tests: PDF parsing, chunking, embeddings, FAISS, LLM abstention
│   └── test_frontend.py       # Integration tests: UI components, callbacks, and end-to-end flow
│
├── data/                      # Data storage
│   └── sample_pdfs/           # Directory for benchmark sample PDFs
│
├── evaluation/                # Benchmarking & QA testing
│   ├── test_questions.csv     # Curated benchmark questions with ground truth
│   └── evaluation_results.csv # Empirical accuracy, citation correctness, & latency logs
│
├── documentation/             # Engineering & architecture specifications
│   ├── problem_analysis.md    # Problem breakdown, challenges, and objectives
│   ├── requirements.md        # Functional and non-functional requirements
│   ├── architecture.md        # System architecture and data flow diagrams
│   └── methodology.md         # Text processing, embedding, and abstention logic
│
└── assets/                    # Static assets, styling, and diagrams
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python**: Version 3.10 to 3.12 (Tested on Python 3.12.10)
- **OS**: Windows, macOS, or Linux
- **Git**

### 2. Setup Virtual Environment
```powershell
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
# Windows:
Copy-Item .env.example .env

# Linux / macOS:
cp .env.example .env
```
Edit `.env` and set your Google Gemini API key:
```ini
GEMINI_API_KEY=AIzaSy...your_actual_key_here
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
GEMINI_MODEL_NAME=gemini-3.6-flash
CHUNK_SIZE=500
CHUNK_OVERLAP=100
SIMILARITY_THRESHOLD=0.35
```
*(Get a free API key at [Google AI Studio](https://aistudio.google.com/)).*

### 5. Launch the Application
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:7860
```

---

## 🧪 Running Automated Tests

Run the full test suite (14 unit and integration tests):
```powershell
python -m unittest discover tests -v
```

All tests run deterministically with in-memory test PDFs and do not require an active Gemini API key to validate indexing, threshold filtering, or abstention logic.

---

## 🖥️ UI Workflow

1. **Upload Documents**: Drag and drop one or more PDF files into the **Document Upload** section in the sidebar.
2. **Process & Index**: Click **⚡ Process & Index**. The system extracts text page by page, segments chunks preserving original page numbers, generates dense embeddings, and builds a FAISS CPU index.
3. **Ask Questions**: Type natural language questions in the query box and click **Ask Question 🔍** or press <kbd>Enter</kbd>.
4. **Inspect Evidence**: The answer displays in the conversational chat with inline citations. The expandable **📋 Evidence & Citations** panel shows verified source snippets, filenames, page numbers, and similarity percentages.
5. **Abstention Verification**: Questions asking for facts not present in the documents trigger the strict abstention mechanism, preventing hallucinations.
