"""
run_ingestion.py
─────────────────
Main ingestion pipeline script.

Usage:
    python run_ingestion.py                     # downloads + ingests NASA PDF
    python run_ingestion.py --pdf path/to.pdf   # use a local PDF
    python run_ingestion.py --recreate          # drop & recreate Qdrant collection

Steps:
    1. Download / locate PDF
    2. Parse section structure (TOC)
    3. Build acronym glossary
    4. Extract text chunks (section-aware, hierarchical)
    5. Extract tables (multi-page merge)
    6. Extract figures / diagrams
    7. Build unified chunk list
    8. Create Qdrant collection (if needed)
    9. Embed + upsert all chunks to Qdrant
"""
import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingestion")


def run_pipeline(pdf_path: Path, recreate_collection: bool = False) -> dict:
    """
    Execute the full ingestion pipeline.
    Returns a summary dict with counts and timing info.
    """
    import config
    from ingestion.acronym_resolver import build_glossary, get_known_acronym_set
    from ingestion.chunk_builder import build_unified_chunks, save_chunks_to_disk
    from ingestion.figure_extractor import extract_figures
    from ingestion.structure_parser import parse_structure
    from ingestion.table_extractor import extract_tables
    from ingestion.text_extractor import extract_all_text_chunks
    from storage.indexer import get_collection_stats, index_chunks
    from storage.qdrant_manager import create_collection

    t0 = time.time()
    summary: dict = {"pdf": str(pdf_path)}

    # ── 1. Parse structure ────────────────────────────────────────────────────
    logger.info("━━━ Step 1/7 — Parsing document structure …")
    nodes = parse_structure(pdf_path)
    summary["sections"] = len(nodes)
    logger.info(f"    → {len(nodes)} sections found")

    # ── 2. Build glossary ─────────────────────────────────────────────────────
    logger.info("━━━ Step 2/7 — Building acronym glossary …")
    glossary = build_glossary(pdf_path)
    known_acronyms = get_known_acronym_set(glossary)
    summary["acronyms"] = len(glossary)
    logger.info(f"    → {len(glossary)} acronyms in glossary")

    # ── 3. Extract text chunks ────────────────────────────────────────────────
    logger.info("━━━ Step 3/7 — Extracting text chunks …")
    text_chunks = extract_all_text_chunks(
        pdf_path=pdf_path,
        nodes=nodes,
        max_tokens=config.MAX_CHUNK_TOKENS,
        overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        known_acronyms=known_acronyms,
    )
    summary["text_chunks"] = len(text_chunks)
    logger.info(f"    → {len(text_chunks)} text chunks")

    # ── 4. Extract tables ─────────────────────────────────────────────────────
    logger.info("━━━ Step 4/7 — Extracting tables …")
    table_chunks = extract_tables(
        pdf_path=pdf_path,
        tables_dir=config.TABLES_DIR,
        section_nodes=nodes,
    )
    summary["table_chunks"] = len(table_chunks)
    logger.info(f"    → {len(table_chunks)} tables")

    # ── 5. Extract figures ────────────────────────────────────────────────────
    logger.info("━━━ Step 5/7 — Extracting figures …")
    figure_chunks = extract_figures(
        pdf_path=pdf_path,
        figures_dir=config.FIGURES_DIR,
        section_nodes=nodes,
    )
    summary["figure_chunks"] = len(figure_chunks)
    logger.info(f"    → {len(figure_chunks)} figures")

    # ── 6. Build unified chunks ───────────────────────────────────────────────
    logger.info("━━━ Step 6/7 — Building unified chunk list …")
    unified_chunks = build_unified_chunks(text_chunks, table_chunks, figure_chunks)
    summary["total_chunks"] = len(unified_chunks)
    save_chunks_to_disk(unified_chunks, config.CHUNKS_DIR)

    # ── 7. Index into Qdrant ──────────────────────────────────────────────────
    logger.info("━━━ Step 7/7 — Indexing into Qdrant Cloud …")
    create_collection(recreate=recreate_collection)
    upserted = index_chunks(unified_chunks)
    summary["upserted"] = upserted

    elapsed = time.time() - t0
    summary["elapsed_sec"] = round(elapsed, 1)

    stats = get_collection_stats()
    summary.update(stats)

    logger.info("━━━ Ingestion complete!")
    logger.info(f"    Total time : {elapsed:.1f}s")
    logger.info(f"    Qdrant points : {stats['points_count']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="NASA Handbook ingestion pipeline")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to PDF (skips download)")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing",
    )
    args = parser.parse_args()

    import config
    from ingestion.pdf_downloader import download_pdf

    pdf_path = args.pdf or config.PDF_PATH
    if not pdf_path.exists():
        logger.info(f"PDF not found at {pdf_path}, downloading …")
        download_pdf(config.NASA_PDF_URL, pdf_path)

    summary = run_pipeline(pdf_path=pdf_path, recreate_collection=args.recreate)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
