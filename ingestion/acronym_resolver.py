"""
ingestion/acronym_resolver.py
──────────────────────────────
Builds an acronym glossary by scanning the PDF for patterns like:
    "Technical Readiness Level (TRL)"
    "Preliminary Design Review (PDR)"

Also includes a hard-coded seed glossary for common NASA SE terms.
The glossary is used to:
  - Tag chunks with the acronyms they contain.
  - Expand acronyms in queries at search time.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Pattern: "Full Term (ACRONYM)"
_ACRO_DEF_RE = re.compile(
    r"([A-Z][a-z][\w\s,/&'-]{4,60})\s*\(([A-Z]{2,8})\)"
)

# Seed glossary for NASA SE common acronyms
_SEED_GLOSSARY: dict[str, str] = {
    "TRL": "Technology Readiness Level",
    "KDP": "Key Decision Point",
    "SRR": "System Requirements Review",
    "PDR": "Preliminary Design Review",
    "CDR": "Critical Design Review",
    "SAR": "System Acceptance Review",
    "FRR": "Flight Readiness Review",
    "DR": "Design Review",
    "MRR": "Mission Requirements Review",
    "ORR": "Operational Readiness Review",
    "PRR": "Production Readiness Review",
    "SDR": "System Design Review",
    "V&V": "Verification and Validation",
    "WBS": "Work Breakdown Structure",
    "PBS": "Product Breakdown Structure",
    "ICD": "Interface Control Document",
    "SRD": "System Requirements Document",
    "SEMP": "Systems Engineering Management Plan",
    "ConOps": "Concept of Operations",
    "MOEs": "Measures of Effectiveness",
    "MOPs": "Measures of Performance",
    "TPMs": "Technical Performance Measures",
    "SE": "Systems Engineering",
    "NASA": "National Aeronautics and Space Administration",
    "PM": "Program/Project Manager",
    "FAR": "Federal Acquisition Regulation",
    "EIA": "Electronic Industries Alliance",
    "ISO": "International Organization for Standardization",
    "RFP": "Request for Proposal",
    "SOW": "Statement of Work",
    "FMEA": "Failure Modes and Effects Analysis",
    "FTA": "Fault Tree Analysis",
    "ITAR": "International Traffic in Arms Regulations",
}


def build_glossary(pdf_path: Path) -> dict[str, str]:
    """
    Scan the full PDF for acronym definitions and return a combined glossary.
    Keys = acronym (uppercase), values = full expansion.
    """
    pdf_path = Path(pdf_path)
    glossary: dict[str, str] = dict(_SEED_GLOSSARY)  # start with seed

    try:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            text = page.get_text("text")
            for m in _ACRO_DEF_RE.finditer(text):
                full_term = m.group(1).strip()
                acronym = m.group(2).strip()
                if acronym not in glossary:
                    glossary[acronym] = full_term
                    logger.debug(f"Discovered acronym: {acronym} → {full_term}")
        doc.close()
    except Exception as e:
        logger.warning(f"Acronym scan failed: {e}")

    logger.info(f"Glossary built: {len(glossary)} acronyms (seed + discovered).")
    return glossary


def expand_query(query: str, glossary: dict[str, str]) -> str:
    """
    Expand acronyms in a user query using the glossary.
    E.g. "What is TRL?" → "What is Technology Readiness Level (TRL)?"
    """
    def _replace(m: re.Match) -> str:
        acr = m.group(0)
        expansion = glossary.get(acr)
        return f"{expansion} ({acr})" if expansion else acr

    return re.sub(r"\b([A-Z]{2,8})\b", _replace, query)


def get_known_acronym_set(glossary: dict[str, str]) -> set[str]:
    """Return just the set of acronym strings for fast membership tests."""
    return set(glossary.keys())
