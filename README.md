# 🚀 NASA Systems Engineering Handbook — QA System

> **Interview Task 2 — Complex Technical Manual QA**  
> Smart PDF ingestion + Qdrant Cloud hybrid search + LLM-powered chatbot

---

## Architecture

```
PDF → Structure Parser → Text / Table / Figure Extractor
    → Acronym Resolver → Cross-Reference Linker
    → Qdrant Cloud (dense + sparse hybrid search)
    → QA Chatbot (GPT-4o-mini + citation engine)
```

## Quick Start

### 1. Clone & Install

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and fill in your credentials:

```bash
copy .env.example .env
```

Required values:
| Variable | Description |
|----------|-------------|
| `QDRANT_URL` | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `OPENAI_API_KEY` | OpenAI API key (for embeddings + LLM) |

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Then in the **📥 Upload & Ingest** tab:
1. Set credentials in the sidebar
2. Choose "Use NASA URL" and click **Download PDF**
3. Click **🚀 Run Full Ingestion Pipeline**

Switch to the **💬 Chatbot** tab to start asking questions.

### 4. Or run ingestion from CLI

```bash
# Download + ingest NASA handbook (first time)
python run_ingestion.py

# Use a local PDF
python run_ingestion.py --pdf path/to/manual.pdf

# Recreate collection (re-ingestion)
python run_ingestion.py --recreate
```

---

## Project Structure

```
practical-task-02/
├── app.py                       # Streamlit UI (Upload + Chatbot tabs)
├── config.py                    # Central config from .env
├── run_ingestion.py             # CLI ingestion pipeline
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── pdf_downloader.py        # Download / save PDF
│   ├── structure_parser.py      # TOC + section hierarchy (PyMuPDF)
│   ├── text_extractor.py        # Section-aware text chunking
│   ├── table_extractor.py       # Multi-page table extraction (pdfplumber)
│   ├── figure_extractor.py      # Figure/chart capture (PyMuPDF)
│   ├── acronym_resolver.py      # Glossary builder + query expansion
│   └── chunk_builder.py         # Unified chunk assembly
│
├── storage/
│   ├── qdrant_manager.py        # Collection creation (dense + sparse vectors)
│   ├── embedder.py              # OpenAI dense + fastembed sparse embeddings
│   └── indexer.py               # Batch upsert to Qdrant
│
├── retrieval/
│   └── hybrid_search.py         # RRF hybrid search + multi-hop expansion
│
├── qa/
│   └── answer_generator.py      # GPT answer + citation formatting
│
├── data/                        # PDF storage
├── outputs/
│   ├── figures/                 # Extracted figure PNGs
│   ├── tables/                  # Extracted table markdowns
│   └── chunks/                  # Debug JSON per chunk
```

---

## Key Design Decisions

### Hybrid Search (Qdrant)
Each chunk is stored with **two named vectors**:
- `dense` — OpenAI `text-embedding-3-small` (1536-dim cosine) for semantic matching
- `sparse` — BM25 sparse vectors via `fastembed` for exact keyword matching

Retrieval uses **Reciprocal Rank Fusion (RRF)** to combine both result lists.

### Hierarchical Chunking
Documents are chunked in three levels: Chapter → Section → Paragraph.
Every chunk carries rich metadata in the Qdrant payload:
- `section_id`, `chapter_num`, `page_numbers`
- `heading_hierarchy` (breadcrumb trail)
- `cross_references` (detected "see Section X.Y" patterns)
- `acronyms_used`

### Multi-hop Cross-Reference Resolution
When a retrieved chunk references another section (e.g., "see Section 4.2"),
the system automatically retrieves that section too and includes it in the context.

### Figure Extraction (No OCR Needed)
The NASA handbook is a digital PDF, so:
- **Captions** are detected via regex (`Figure 6-3 …`)
- The **page region** around the caption is rendered as a high-res PNG
- **Embedded raster images** (diagrams ≥200×200px) are extracted directly

### Table Extraction
`pdfplumber` detects tables per page. Multi-page tables are merged by detecting
same-column-count continuation pages. Tables are stored as **Markdown + JSON**
for both readable embedding and structured retrieval.
