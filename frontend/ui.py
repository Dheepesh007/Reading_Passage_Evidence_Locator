"""
Professional Gradio Blocks UI for Reading Passage Evidence Locator (PS 77).

A premium, modern Academic AI Platform interface with:
- Custom CSS design system (light/dark themes, navy/indigo accents)
- Sidebar with branding and navigation
- Drag-and-drop PDF upload with status cards
- Conversational chat with evidence citations
- Expandable source evidence panels
- Loading indicators and empty states
"""

import gradio as gr
from typing import Optional

# ---------------------------------------------------------------------------
# DESIGN SYSTEM – Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ========================================================================
   READING PASSAGE EVIDENCE LOCATOR — DESIGN SYSTEM
   Premium Academic AI Platform Theme
   ======================================================================== */

/* ---------- CSS Variables (Light Theme) ---------- */
:root {
    --rpel-primary: #1e3a5f;
    --rpel-primary-light: #2a5298;
    --rpel-primary-dark: #0f1f36;
    --rpel-accent: #4f46e5;
    --rpel-accent-light: #6366f1;
    --rpel-accent-soft: #eef2ff;
    --rpel-success: #059669;
    --rpel-success-light: #d1fae5;
    --rpel-warning: #d97706;
    --rpel-warning-light: #fef3c7;
    --rpel-danger: #dc2626;
    --rpel-danger-light: #fee2e2;
    --rpel-bg: #f8fafc;
    --rpel-bg-card: #ffffff;
    --rpel-bg-sidebar: linear-gradient(180deg, #1e3a5f 0%, #0f1f36 100%);
    --rpel-bg-input: #f1f5f9;
    --rpel-bg-hover: #e2e8f0;
    --rpel-text: #1e293b;
    --rpel-text-secondary: #64748b;
    --rpel-text-muted: #94a3b8;
    --rpel-text-on-primary: #ffffff;
    --rpel-border: #e2e8f0;
    --rpel-border-focus: #4f46e5;
    --rpel-radius: 12px;
    --rpel-radius-sm: 8px;
    --rpel-radius-lg: 16px;
    --rpel-shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --rpel-shadow: 0 4px 12px rgba(0,0,0,.07), 0 1px 3px rgba(0,0,0,.05);
    --rpel-shadow-lg: 0 10px 25px rgba(0,0,0,.08), 0 4px 10px rgba(0,0,0,.04);
    --rpel-transition: all .2s ease;
    --rpel-font: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ---------- Dark Theme Override ---------- */
.dark {
    --rpel-bg: #0f172a;
    --rpel-bg-card: #1e293b;
    --rpel-bg-sidebar: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    --rpel-bg-input: #334155;
    --rpel-bg-hover: #334155;
    --rpel-text: #f1f5f9;
    --rpel-text-secondary: #94a3b8;
    --rpel-text-muted: #64748b;
    --rpel-border: #334155;
    --rpel-accent-soft: #1e1b4b;
    --rpel-success-light: #064e3b;
    --rpel-warning-light: #451a03;
    --rpel-danger-light: #450a0a;
    --rpel-shadow-sm: 0 1px 3px rgba(0,0,0,.2);
    --rpel-shadow: 0 4px 12px rgba(0,0,0,.25);
    --rpel-shadow-lg: 0 10px 25px rgba(0,0,0,.35);
}

/* ---------- Global Reset & Base ---------- */
.gradio-container {
    font-family: var(--rpel-font) !important;
    background: var(--rpel-bg) !important;
    max-width: 100% !important;
    padding: 0 !important;
}

/* ---------- Header / Hero Section ---------- */
.rpel-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a5298 50%, #4f46e5 100%);
    padding: 28px 36px;
    border-radius: 0 0 var(--rpel-radius-lg) var(--rpel-radius-lg);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.rpel-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,.06) 0%, transparent 70%);
    border-radius: 50%;
}
.rpel-header h1 {
    color: #ffffff !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.02em;
}
.rpel-header p {
    color: rgba(255,255,255,.75) !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
    font-weight: 400;
}
.rpel-badge {
    display: inline-block;
    background: rgba(255,255,255,.15);
    backdrop-filter: blur(10px);
    color: #ffffff;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ---------- Card Styling ---------- */
.rpel-card {
    background: var(--rpel-bg-card) !important;
    border: 1px solid var(--rpel-border) !important;
    border-radius: var(--rpel-radius) !important;
    padding: 20px !important;
    box-shadow: var(--rpel-shadow-sm) !important;
    transition: var(--rpel-transition) !important;
}
.rpel-card:hover {
    box-shadow: var(--rpel-shadow) !important;
}
.rpel-card-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--rpel-primary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---------- Sidebar Panel ---------- */
.rpel-sidebar {
    background: var(--rpel-bg-card) !important;
    border: 1px solid var(--rpel-border) !important;
    border-radius: var(--rpel-radius) !important;
    box-shadow: var(--rpel-shadow) !important;
    padding: 0 !important;
    overflow: hidden;
}
.rpel-sidebar-brand {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a5298 100%);
    padding: 20px;
    text-align: center;
}
.rpel-sidebar-brand h3 {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    margin: 0 0 2px 0 !important;
}
.rpel-sidebar-brand p {
    color: rgba(255,255,255,.65) !important;
    font-size: 0.75rem !important;
    margin: 0 !important;
}
.rpel-sidebar-section {
    padding: 16px 20px;
    border-bottom: 1px solid var(--rpel-border);
}
.rpel-sidebar-section:last-child { border-bottom: none; }
.rpel-sidebar-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--rpel-text-muted);
    margin-bottom: 10px;
}

/* ---------- Upload Zone ---------- */
.rpel-upload-zone {
    border: 2px dashed var(--rpel-border) !important;
    border-radius: var(--rpel-radius) !important;
    background: var(--rpel-bg-input) !important;
    transition: var(--rpel-transition) !important;
    min-height: 100px !important;
}
.rpel-upload-zone:hover {
    border-color: var(--rpel-accent) !important;
    background: var(--rpel-accent-soft) !important;
}

/* ---------- Status Indicator Styling ---------- */
.rpel-status-ready {
    background: var(--rpel-success-light) !important;
    color: var(--rpel-success) !important;
    border: 1px solid var(--rpel-success) !important;
    border-radius: var(--rpel-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 14px !important;
}
.rpel-status-idle {
    background: var(--rpel-bg-input) !important;
    color: var(--rpel-text-secondary) !important;
    border: 1px solid var(--rpel-border) !important;
    border-radius: var(--rpel-radius-sm) !important;
    font-size: 0.82rem !important;
    padding: 8px 14px !important;
}
.rpel-status-error {
    background: var(--rpel-danger-light) !important;
    color: var(--rpel-danger) !important;
    border: 1px solid var(--rpel-danger) !important;
    border-radius: var(--rpel-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 14px !important;
}

/* ---------- Stat Metric Cards ---------- */
.rpel-stat {
    text-align: center;
    padding: 12px 8px;
}
.rpel-stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--rpel-accent);
    line-height: 1.2;
}
.rpel-stat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--rpel-text-muted);
    margin-top: 2px;
}

/* ---------- Buttons ---------- */
.rpel-btn-primary {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--rpel-radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    transition: var(--rpel-transition) !important;
    box-shadow: 0 2px 8px rgba(79,70,229,.25) !important;
    cursor: pointer !important;
}
.rpel-btn-primary:hover {
    box-shadow: 0 4px 16px rgba(79,70,229,.35) !important;
    transform: translateY(-1px) !important;
}
.rpel-btn-secondary {
    background: var(--rpel-bg-input) !important;
    color: var(--rpel-text) !important;
    border: 1px solid var(--rpel-border) !important;
    border-radius: var(--rpel-radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    transition: var(--rpel-transition) !important;
    cursor: pointer !important;
}
.rpel-btn-secondary:hover {
    background: var(--rpel-bg-hover) !important;
}
.rpel-btn-danger {
    background: transparent !important;
    color: var(--rpel-danger) !important;
    border: 1px solid var(--rpel-danger) !important;
    border-radius: var(--rpel-radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    transition: var(--rpel-transition) !important;
    cursor: pointer !important;
}
.rpel-btn-danger:hover {
    background: var(--rpel-danger-light) !important;
}

/* ---------- Chat Container ---------- */
.rpel-chat-container {
    background: var(--rpel-bg-card) !important;
    border: 1px solid var(--rpel-border) !important;
    border-radius: var(--rpel-radius) !important;
    box-shadow: var(--rpel-shadow) !important;
    overflow: hidden;
}

/* ---------- Chatbot Messages ---------- */
.rpel-chatbot {
    border: none !important;
    background: transparent !important;
}
.rpel-chatbot .message {
    border-radius: var(--rpel-radius) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}

/* ---------- Input Bar ---------- */
.rpel-input-row {
    background: var(--rpel-bg-card);
    border-top: 1px solid var(--rpel-border);
    padding: 14px 16px;
    border-radius: 0 0 var(--rpel-radius) var(--rpel-radius);
}
.rpel-textbox textarea,
.rpel-textbox input {
    border: 1.5px solid var(--rpel-border) !important;
    border-radius: var(--rpel-radius-sm) !important;
    background: var(--rpel-bg-input) !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
    transition: var(--rpel-transition) !important;
}
.rpel-textbox textarea:focus,
.rpel-textbox input:focus {
    border-color: var(--rpel-border-focus) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.12) !important;
    outline: none !important;
}

/* ---------- Evidence Accordion ---------- */
.rpel-evidence-panel {
    background: var(--rpel-bg-card) !important;
    border: 1px solid var(--rpel-border) !important;
    border-radius: var(--rpel-radius) !important;
    box-shadow: var(--rpel-shadow-sm) !important;
    margin-top: 16px;
}

/* ---------- Evidence Source Card ---------- */
.rpel-source-card {
    padding: 14px 18px;
    background: var(--rpel-accent-soft);
    border-left: 3px solid var(--rpel-accent);
    border-radius: 0 var(--rpel-radius-sm) var(--rpel-radius-sm) 0;
    margin: 8px 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.rpel-source-tag {
    display: inline-block;
    background: var(--rpel-accent);
    color: #ffffff;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 6px;
    letter-spacing: 0.03em;
}
.rpel-score-badge {
    display: inline-block;
    background: var(--rpel-success-light);
    color: var(--rpel-success);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 700;
}

/* ---------- Empty State ---------- */
.rpel-empty-state {
    text-align: center;
    padding: 48px 24px;
    color: var(--rpel-text-muted);
}
.rpel-empty-state .icon {
    font-size: 2.5rem;
    margin-bottom: 12px;
    opacity: 0.5;
}
.rpel-empty-state h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--rpel-text-secondary) !important;
    margin: 0 0 6px 0 !important;
}
.rpel-empty-state p {
    font-size: 0.82rem !important;
    color: var(--rpel-text-muted) !important;
    margin: 0 !important;
}

/* ---------- Footer ---------- */
.rpel-footer {
    text-align: center;
    padding: 20px;
    color: var(--rpel-text-muted);
    font-size: 0.75rem;
}
.rpel-footer a {
    color: var(--rpel-accent);
    text-decoration: none;
}

/* ---------- Utility ---------- */
.rpel-divider {
    border: none;
    border-top: 1px solid var(--rpel-border);
    margin: 16px 0;
}
.rpel-mb-0 { margin-bottom: 0 !important; }
.rpel-mt-sm { margin-top: 8px !important; }
.rpel-hide-label label { display: none !important; }

/* Gradio component overrides */
.gradio-container .prose { font-family: var(--rpel-font) !important; }
#rpel-app { min-height: 100vh; }
"""

# ---------------------------------------------------------------------------
# HTML FRAGMENTS
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div class="rpel-header">
    <div class="rpel-badge">PS 77 · Hackathon Project</div>
    <h1>📖 Reading Passage Evidence Locator</h1>
    <p>Evidence-grounded document Q&A with strict page provenance — powered by Google Gemini &amp; FAISS</p>
</div>
"""

SIDEBAR_BRAND_HTML = """
<div class="rpel-sidebar-brand">
    <h3>📖 RPEL</h3>
    <p>Evidence Locator v1.0</p>
</div>
"""

EMPTY_CHAT_HTML = """
<div class="rpel-empty-state">
    <div class="icon">💬</div>
    <h3>Ask a Question</h3>
    <p>Upload PDF documents and ask questions to receive<br>evidence-grounded answers with page citations.</p>
</div>
"""

EMPTY_EVIDENCE_HTML = """
<div class="rpel-empty-state" style="padding: 32px 20px;">
    <div class="icon">📑</div>
    <h3>No Evidence Retrieved Yet</h3>
    <p>Source passages, page numbers, and similarity scores<br>will appear here after each query.</p>
</div>
"""

NO_DOCS_HTML = """
<div class="rpel-empty-state" style="padding: 24px 16px;">
    <div class="icon">📄</div>
    <h3>No Documents Loaded</h3>
    <p>Upload one or more PDF files to begin.</p>
</div>
"""

# ---------------------------------------------------------------------------
# EVIDENCE FORMATTING
# ---------------------------------------------------------------------------

def format_evidence_html(sources: list, abstained: bool = False) -> str:
    """Renders retrieved evidence passages as styled HTML source cards."""
    if abstained:
        return """
<div class="rpel-source-card" style="border-left-color: var(--rpel-warning); background: var(--rpel-warning-light); padding: 18px 20px;">
    <div style="font-weight: 700; color: var(--rpel-warning); font-size: 0.95rem; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
        <span>⚠️</span>
        <span>Insufficient Grounding Evidence (Abstained)</span>
    </div>
    <div style="color: var(--rpel-text); font-size: 0.85rem; line-height: 1.6;">
        The system abstained from answering because the uploaded document(s) do not contain sufficient evidence 
        matching this query with high confidence (similarity threshold &ge; 0.35). 
        <br><br>
        <em>Grounded RAG Guarantee:</em> Rather than fabricating or guessing citations, the locator strictly halts when verifiable source sentences cannot be proven.
    </div>
</div>
"""

    if not sources:
        return EMPTY_EVIDENCE_HTML

    cards = []
    for i, src in enumerate(sources, 1):
        doc_name = src.get("document_name", "Unknown")
        page_num = src.get("page_number", "?")
        text = src.get("text", "")
        score = src.get("similarity_score", 0.0)

        score_pct = f"{score * 100:.1f}%"
        card = f"""
<div class="rpel-source-card">
    <div style="margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center;">
        <span class="rpel-source-tag">Source {i}</span>
        <span class="rpel-source-tag" style="background: var(--rpel-primary);">📄 {doc_name}</span>
        <span class="rpel-source-tag" style="background: #0284c7;">Page {page_num}</span>
        <span class="rpel-score-badge">⚡ {score_pct} relevance</span>
    </div>
    <div style="color: var(--rpel-text); font-style: italic; font-size: 0.86rem; line-height: 1.6;">"{text}"</div>
</div>"""
        cards.append(card)

    header = f"""<div class="rpel-card-header">📋 Retrieved Evidence — {len(sources)} source passage(s) verified</div>"""
    return header + "\n".join(cards)


def format_status_html(status: str, details: str = "") -> str:
    """Returns styled status indicator HTML."""
    if status == "ready":
        cls = "rpel-status-ready"
        icon = "✅"
    elif status == "error":
        cls = "rpel-status-error"
        icon = "❌"
    else:
        cls = "rpel-status-idle"
        icon = "⏳"
    return f'<div class="{cls}">{icon} {details}</div>'


def format_doc_list_html(doc_names: list, stats: dict = None) -> str:
    """Renders a list of indexed documents as styled HTML."""
    if not doc_names:
        return NO_DOCS_HTML

    items = ""
    for name in doc_names:
        items += f"""
<div style="display:flex; align-items:center; gap:8px; padding:8px 12px;
            background:var(--rpel-accent-soft); border-radius:var(--rpel-radius-sm);
            margin-bottom:6px; font-size:0.82rem;">
    <span>📄</span>
    <span style="flex:1; font-weight:500; color:var(--rpel-text);">{name}</span>
    <span style="color:var(--rpel-success); font-size:0.75rem;">✓ Indexed</span>
</div>"""

    stat_row = ""
    if stats:
        stat_row = f"""
<div style="display:flex; gap:12px; margin-top:12px;">
    <div class="rpel-stat" style="flex:1; background:var(--rpel-accent-soft); border-radius:var(--rpel-radius-sm);">
        <div class="rpel-stat-value">{stats.get('total_pages', 0)}</div>
        <div class="rpel-stat-label">Pages</div>
    </div>
    <div class="rpel-stat" style="flex:1; background:var(--rpel-accent-soft); border-radius:var(--rpel-radius-sm);">
        <div class="rpel-stat-value">{stats.get('total_chunks', 0)}</div>
        <div class="rpel-stat-label">Chunks</div>
    </div>
    <div class="rpel-stat" style="flex:1; background:var(--rpel-success-light); border-radius:var(--rpel-radius-sm);">
        <div class="rpel-stat-value">{stats.get('elapsed_time', 0)}s</div>
        <div class="rpel-stat-label">Index Time</div>
    </div>
</div>"""

    return f'<div class="rpel-sidebar-label">📂 Indexed Documents</div>{items}{stat_row}'


# ---------------------------------------------------------------------------
# BUILD UI
# ---------------------------------------------------------------------------

def build_ui(
    process_callback=None,
    ask_callback=None,
    clear_callback=None
):
    """
    Constructs the full professional Gradio Blocks application.

    Args:
        process_callback:  fn(files) -> (status_html, doc_list_html)
        ask_callback:      fn(question, chat_history) -> (chat_history, evidence_html)
        clear_callback:    fn() -> (chat_history, evidence_html, status_html, doc_list_html)
    """

    # In Gradio 6.x, theme and css are passed to launch(), not Blocks()
    rpel_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        title="RPEL — Reading Passage Evidence Locator",
        fill_height=True,
        elem_id="rpel-app"
    ) as demo:
        # Store theme/css for launch()
        demo._rpel_theme = rpel_theme
        demo._rpel_css = CUSTOM_CSS

        # ===== HEADER =====
        gr.HTML(HEADER_HTML)

        # ===== MAIN LAYOUT: SIDEBAR + CONTENT =====
        with gr.Row(equal_height=False):

            # ---- LEFT SIDEBAR ----
            with gr.Column(scale=1, min_width=300):

                # Sidebar Card
                with gr.Group(elem_classes="rpel-sidebar"):

                    gr.HTML(SIDEBAR_BRAND_HTML)

                    # Upload Section
                    with gr.Group(elem_classes="rpel-sidebar-section"):
                        gr.HTML('<div class="rpel-sidebar-label">📤 Document Upload</div>')
                        pdf_input = gr.File(
                            label="Drop PDFs here or click to browse",
                            file_count="multiple",
                            file_types=[".pdf"],
                            elem_classes="rpel-upload-zone rpel-hide-label",
                            height=120,
                        )
                        with gr.Row():
                            process_btn = gr.Button(
                                "⚡ Process & Index",
                                variant="primary",
                                elem_classes="rpel-btn-primary",
                                size="sm"
                            )
                            clear_all_btn = gr.Button(
                                "🗑 Clear All",
                                variant="secondary",
                                elem_classes="rpel-btn-danger",
                                size="sm"
                            )

                    # Status Section
                    with gr.Group(elem_classes="rpel-sidebar-section"):
                        gr.HTML('<div class="rpel-sidebar-label">📊 System Status</div>')
                        status_display = gr.HTML(
                            value=format_status_html("idle", "Waiting for document upload...")
                        )

                    # Indexed Documents Section
                    with gr.Group(elem_classes="rpel-sidebar-section"):
                        doc_list_display = gr.HTML(value=NO_DOCS_HTML)

                    # Info Section
                    with gr.Group(elem_classes="rpel-sidebar-section"):
                        gr.HTML("""
                        <div class="rpel-sidebar-label">ℹ️ How It Works</div>
                        <div style="font-size:0.78rem; color:var(--rpel-text-secondary); line-height:1.7;">
                            <strong>1.</strong> Upload one or more PDF documents<br>
                            <strong>2.</strong> Click <em>Process & Index</em> to analyze<br>
                            <strong>3.</strong> Ask questions in natural language<br>
                            <strong>4.</strong> Get answers with page citations<br><br>
                            <span style="color:var(--rpel-text-muted); font-size:0.72rem;">
                            Only answers supported by document evidence are shown.
                            If no evidence is found, the system will abstain.</span>
                        </div>
                        """)

            # ---- RIGHT CONTENT: CHAT + EVIDENCE ----
            with gr.Column(scale=3, min_width=500):

                # Chat Container
                with gr.Group(elem_classes="rpel-chat-container"):

                    # Chat Header
                    gr.HTML("""
                    <div style="padding:14px 20px; border-bottom:1px solid var(--rpel-border);
                                display:flex; align-items:center; justify-content:space-between;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="font-size:1.2rem;">🤖</span>
                            <div>
                                <div style="font-weight:600; font-size:0.9rem; color:var(--rpel-text);">
                                    Evidence Q&A Assistant
                                </div>
                                <div style="font-size:0.72rem; color:var(--rpel-text-muted);">
                                    Powered by Google Gemini · Grounded in your documents
                                </div>
                            </div>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span style="width:8px; height:8px; border-radius:50%;
                                         background:var(--rpel-success); display:inline-block;"></span>
                            <span style="font-size:0.72rem; color:var(--rpel-success); font-weight:600;">Online</span>
                        </div>
                    </div>
                    """)

                    # Chatbot
                    chatbot = gr.Chatbot(
                        value=[],
                        height=440,
                        elem_classes="rpel-chatbot",
                        show_label=False,
                        placeholder=EMPTY_CHAT_HTML,
                    )

                    # Input Bar
                    with gr.Row(elem_classes="rpel-input-row"):
                        msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask a question about your uploaded documents...",
                            elem_classes="rpel-textbox",
                            scale=5,
                            container=False,
                            lines=1,
                            max_lines=3,
                        )
                        submit_btn = gr.Button(
                            "Ask Question 🔍",
                            variant="primary",
                            elem_classes="rpel-btn-primary",
                            scale=1,
                            min_width=130,
                            size="sm",
                        )

                # Evidence Panel
                with gr.Accordion(
                    "📋 Evidence & Citations",
                    open=True,
                    elem_classes="rpel-evidence-panel"
                ):
                    evidence_display = gr.HTML(value=EMPTY_EVIDENCE_HTML)

        # ===== FOOTER =====
        gr.HTML("""
        <div class="rpel-footer">
            <strong>Reading Passage Evidence Locator</strong> · Problem Statement 77<br>
            Built with Gradio · Google Gemini · FAISS · Sentence Transformers
        </div>
        """)

        # ===== EVENT WIRING =====
        if process_callback:
            process_btn.click(
                fn=process_callback,
                inputs=[pdf_input],
                outputs=[status_display, doc_list_display],
                show_progress="full",
            )

        if ask_callback:
            # Submit via button
            submit_btn.click(
                fn=ask_callback,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, evidence_display, msg_input],
                show_progress="minimal",
            )
            # Submit via Enter key
            msg_input.submit(
                fn=ask_callback,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, evidence_display, msg_input],
                show_progress="minimal",
            )

        if clear_callback:
            clear_all_btn.click(
                fn=clear_callback,
                inputs=[],
                outputs=[chatbot, evidence_display, status_display, doc_list_display],
            )

    return demo
