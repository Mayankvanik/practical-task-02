"""
app.py — Streamlit UI for the NASA Handbook QA System
──────────────────────────────────────────────────────
Two tabs:
  📥  Upload & Ingest  — upload a PDF (or use the NASA URL) and run the full ingestion pipeline
  💬  Chatbot          — ask questions with source citations and figure previews
"""
import os
import sys
import time
import logging
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASA SE Handbook — QA System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .qa-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .citation-card {
        background: rgba(100,130,255,0.10);
        border-left: 4px solid #6482ff;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.88rem;
    }
    .cross-ref-badge {
        display: inline-block;
        background: rgba(255,165,0,0.2);
        border: 1px solid orange;
        color: orange;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.75rem;
        margin-left: 6px;
    }
    .chunk-type-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .badge-text   { background:#3b5bff22; color:#7b9dff; border:1px solid #3b5bff44; }
    .badge-table  { background:#10b98122; color:#34d399; border:1px solid #10b98144; }
    .badge-figure { background:#f5973522; color:#fbbf24; border:1px solid #f5973544; }

    /* Answer box */
    .answer-box {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.10);
        white-space: pre-wrap;
        line-height: 1.7;
    }

    /* Stat metric tiles */
    .metric-tile {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .metric-tile h2 { margin: 0; font-size: 2rem; color: #7b9dff; }
    .metric-tile p  { margin: 0; font-size: 0.85rem; color: #aaa; }

    /* Message bubbles */
    .user-bubble {
        background: linear-gradient(90deg, #3b5bff44, #5b3bff22);
        border-radius: 16px 16px 4px 16px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(100,130,255,0.25);
    }
    .assistant-bubble {
        background: rgba(255,255,255,0.05);
        border-radius: 16px 16px 16px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.10);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b5bff, #7b3bff);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-weight: 500; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 NASA SE Handbook QA")
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    qdrant_url = st.text_input(
        "Qdrant Cloud URL",
        value=os.getenv("QDRANT_URL", ""),
        type="default",
        help="e.g. https://xxxx.cloud.qdrant.io:6333",
    )
    qdrant_key = st.text_input(
        "Qdrant API Key",
        value=os.getenv("QDRANT_API_KEY", ""),
        type="password",
    )
    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    collection_name = st.text_input(
        "Collection Name",
        value=os.getenv("QDRANT_COLLECTION", "nasa_handbook"),
    )

    if st.button("💾 Save Config"):
        os.environ["QDRANT_URL"] = qdrant_url
        os.environ["QDRANT_API_KEY"] = qdrant_key
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["QDRANT_COLLECTION"] = collection_name
        st.success("Config saved for this session!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Search Settings")
    top_k = st.slider("Top K results", 3, 20, 8)
    use_multi_hop = st.toggle("Multi-hop cross-reference", value=True)
    filter_type = st.selectbox(
        "Filter chunk type",
        ["All", "text", "table", "figure"],
    )

    st.markdown("---")
    st.caption("NASA Systems Engineering Handbook SP-2016-6105 Rev2 · Public Domain")


# ── Validate credentials are set ─────────────────────────────────────────────
def _creds_ok() -> bool:
    return all(
        [
            os.getenv("QDRANT_URL"),
            os.getenv("QDRANT_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ingest, tab_chat = st.tabs(["📥 Upload & Ingest", "💬 Chatbot"])


# ════════════════════════════════════════════════════════════════
#  TAB 1 — UPLOAD & INGEST
# ════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown(
        "<h1 style='margin-bottom:0.2rem'>📥 PDF Ingestion Pipeline</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Upload a technical PDF or use the NASA SE Handbook URL. "
        "The system will extract text, tables, and figures, then store everything in Qdrant Cloud."
    )
    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 📄 PDF Source")
        source_mode = st.radio(
            "Choose source",
            ["Upload PDF", "Use NASA URL", "Use existing local file"],
            horizontal=True,
        )

        pdf_path: Path | None = None

        if source_mode == "Upload PDF":
            uploaded = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                help="Max ~500 MB. Large PDFs may take several minutes.",
            )
            if uploaded:
                dest = Path("data") / uploaded.name
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(uploaded.read())
                pdf_path = dest
                st.success(f"✅ Saved: `{dest}` ({dest.stat().st_size / 1_000_000:.1f} MB)")

        elif source_mode == "Use NASA URL":
            nasa_url = st.text_input(
                "PDF URL",
                value="https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf",
            )
            pdf_path = Path("data/nasa_handbook.pdf")
            if st.button("⬇️ Download PDF"):
                if not _creds_ok():
                    st.error("⚠️ Please set Qdrant + OpenAI credentials in the sidebar first.")
                else:
                    from ingestion.pdf_downloader import download_pdf
                    with st.spinner("Downloading PDF …"):
                        download_pdf(nasa_url, pdf_path)
                    st.success(f"Downloaded to `{pdf_path}`")

        else:
            pdf_path_str = st.text_input("Local PDF path", value="data/nasa_handbook.pdf")
            pdf_path = Path(pdf_path_str)

    with col_right:
        st.markdown("#### ⚙️ Ingestion Options")
        recreate_col = st.toggle(
            "Recreate Qdrant collection",
            value=False,
            help="WARNING: This deletes and recreates the collection. Use when re-ingesting.",
        )
        show_chunks_preview = st.toggle("Show chunk preview after ingestion", value=True)

    st.markdown("---")

    if st.button("🚀 Run Full Ingestion Pipeline", use_container_width=True):
        if not _creds_ok():
            st.error("⚠️ Set Qdrant URL, Qdrant API Key, and OpenAI API Key in the sidebar!")
        elif pdf_path is None or not pdf_path.exists():
            st.error(f"⚠️ PDF not found at `{pdf_path}`. Please upload or download first.")
        else:
            # Reload env-based config with sidebar values
            import importlib
            import config as cfg_module
            importlib.reload(cfg_module)

            progress = st.progress(0, "Starting pipeline …")
            log_area = st.empty()
            log_lines: list[str] = []

            class StreamlitHandler(logging.Handler):
                def emit(self, record):
                    log_lines.append(self.format(record))
                    log_area.code("\n".join(log_lines[-30:]), language="")

            handler = StreamlitHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s — %(message)s", "%H:%M:%S"))
            logging.getLogger().addHandler(handler)

            try:
                progress.progress(5, "Parsing structure …")
                from run_ingestion import run_pipeline

                # Run synchronously (blocking) — Streamlit handles this fine
                with st.spinner("Ingestion in progress (may take 5–20 min for large PDFs) …"):
                    summary = run_pipeline(
                        pdf_path=pdf_path,
                        recreate_collection=recreate_col,
                    )

                progress.progress(100, "Done!")
                st.success("✅ Ingestion complete!")

                # ── Summary metrics ───────────────────────────────────────────
                st.markdown("### 📊 Ingestion Summary")
                c1, c2, c3, c4, c5 = st.columns(5)
                metrics = [
                    (c1, "📝 Text Chunks", summary.get("text_chunks", 0)),
                    (c2, "📊 Tables", summary.get("table_chunks", 0)),
                    (c3, "🖼️ Figures", summary.get("figure_chunks", 0)),
                    (c4, "🔤 Acronyms", summary.get("acronyms", 0)),
                    (c5, "✅ Upserted", summary.get("upserted", 0)),
                ]
                for col, label, val in metrics:
                    with col:
                        st.markdown(
                            f'<div class="metric-tile"><h2>{val}</h2><p>{label}</p></div>',
                            unsafe_allow_html=True,
                        )

                st.info(
                    f"⏱️ Elapsed: {summary.get('elapsed_sec', 0)}s  •  "
                    f"Qdrant points: {summary.get('points_count', '?')}  •  "
                    f"Collection: `{summary.get('collection', '?')}`"
                )

            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")
                st.exception(e)
            finally:
                logging.getLogger().removeHandler(handler)

    # ── Collection status ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📡 Qdrant Collection Status")
    if st.button("🔄 Refresh Status"):
        if not _creds_ok():
            st.warning("Set credentials first.")
        else:
            try:
                import importlib, config as cfg_module
                importlib.reload(cfg_module)
                from storage.indexer import get_collection_stats
                stats = get_collection_stats()
                st.json(stats)
            except Exception as e:
                st.error(f"Cannot reach Qdrant: {e}")


# ════════════════════════════════════════════════════════════════
#  TAB 2 — CHATBOT
# ════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(
        "<h1 style='margin-bottom:0.2rem'>💬 Ask the NASA SE Handbook</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Ask any question about systems engineering, lifecycle phases, reviews, processes, or diagrams. "
        "The system retrieves relevant sections using **hybrid semantic + keyword search**."
    )
    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Example questions
    with st.expander("💡 Example Questions", expanded=False):
        examples = [
            "What is the Vee Model and how does it relate to systems engineering?",
            "What are the entry criteria for PDR (Preliminary Design Review)?",
            "How does risk management feed into technical reviews?",
            "What does the systems engineering process flow look like?",
            "What is Technology Readiness Level (TRL) and how is it assessed?",
            "What should a verification plan include?",
            "Explain the relationship between requirements and verification.",
            "What are the Key Decision Points (KDPs) in the NASA lifecycle?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:30]}"):
                st.session_state["prefill_query"] = ex
                st.rerun()

    # ── Chat display ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">👤 <b>You:</b><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">🤖 <b>Assistant:</b><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Show citations
                if msg.get("citations"):
                    with st.expander(f"📚 Sources ({len(msg['citations'])} references)", expanded=False):
                        for cit in msg["citations"]:
                            badge_class = {
                                "text": "badge-text",
                                "table": "badge-table",
                                "figure": "badge-figure",
                            }.get(cit["chunk_type"], "badge-text")
                            cross_ref_html = (
                                '<span class="cross-ref-badge">via cross-ref</span>'
                                if cit.get("via_cross_ref")
                                else ""
                            )
                            pages_str = ", ".join(map(str, cit.get("pages", [])))
                            st.markdown(
                                f"""<div class="citation-card">
                                    <b>Source {cit['source_num']}</b>
                                    <span class="chunk-type-badge {badge_class}">{cit['chunk_type']}</span>
                                    {cross_ref_html}<br>
                                    📌 Section <b>{cit['section_id']}</b> — {cit['section_title']}<br>
                                    📖 Chapter {cit.get('chapter_num', '?')}: {cit.get('chapter_title', '')}<br>
                                    📄 Pages: {pages_str}
                                </div>""",
                                unsafe_allow_html=True,
                            )

                # Show figures if any
                if msg.get("figures"):
                    with st.expander(f"🖼️ Related Figures ({len(msg['figures'])})", expanded=False):
                        fig_cols = st.columns(min(3, len(msg["figures"])))
                        for i, fig in enumerate(msg["figures"]):
                            fig_path = Path(fig.get("figure_path", ""))
                            with fig_cols[i % 3]:
                                if fig_path.exists():
                                    st.image(str(fig_path), caption=fig.get("caption", ""), use_container_width=True)
                                else:
                                    st.markdown(f"🖼️ {fig.get('caption', 'Figure')} *(file not found)*")

    st.markdown("---")

    # ── Input row ─────────────────────────────────────────────────────────────
    prefill = st.session_state.pop("prefill_query", "")
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        query = st.text_input(
            "Your question",
            value=prefill,
            placeholder="e.g. What are the entry criteria for CDR?",
            label_visibility="collapsed",
        )
    with col_btn:
        ask_clicked = st.button("Ask 🚀", use_container_width=True)

    # ── Process query ─────────────────────────────────────────────────────────
    if ask_clicked and query.strip():
        if not _creds_ok():
            st.error("⚠️ Please configure credentials in the sidebar first.")
        else:
            import importlib, config as cfg_module
            importlib.reload(cfg_module)

            with st.spinner("Searching knowledge base …"):
                try:
                    from ingestion.acronym_resolver import expand_query
                    from retrieval.hybrid_search import search_with_cross_refs
                    from qa.answer_generator import generate_answer

                    # Expand acronyms in query (using seed glossary)
                    from ingestion.acronym_resolver import _SEED_GLOSSARY
                    expanded_query = expand_query(query, _SEED_GLOSSARY)

                    # Retrieve
                    ft = None if filter_type == "All" else filter_type
                    hits = search_with_cross_refs(
                        query=expanded_query,
                        top_k=top_k,
                        expand_refs=use_multi_hop,
                        filter_chunk_type=ft,
                    )

                    # Generate answer
                    result = generate_answer(query=query, retrieved_hits=hits)

                    # Append to history
                    st.session_state.chat_history.append(
                        {"role": "user", "content": query}
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "citations": result["citations"],
                            "figures": result["figures"],
                        }
                    )
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.exception(e)

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()
