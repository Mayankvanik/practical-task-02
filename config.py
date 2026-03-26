"""
config.py — Central configuration for the NASA Handbook QA System.
Loads all settings from environment variables (.env file).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.environ["QDRANT_URL"]
QDRANT_API_KEY: str = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "nasa_handbook")

# ─── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))

# ─── PDF / Data paths ─────────────────────────────────────────────────────────
NASA_PDF_URL: str = os.getenv(
    "NASA_PDF_URL",
    "https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf",
)
PDF_PATH: Path = Path(os.getenv("PDF_PATH", "data/nasa_handbook.pdf"))
FIGURES_DIR: Path = Path(os.getenv("FIGURES_DIR", "outputs/figures"))
TABLES_DIR: Path = Path(os.getenv("TABLES_DIR", "outputs/tables"))
CHUNKS_DIR: Path = Path(os.getenv("CHUNKS_DIR", "outputs/chunks"))

# ─── Chunking ─────────────────────────────────────────────────────────────────
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "600"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))

# ─── Ensure output directories exist ─────────────────────────────────────────
for _d in [PDF_PATH.parent, FIGURES_DIR, TABLES_DIR, CHUNKS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
