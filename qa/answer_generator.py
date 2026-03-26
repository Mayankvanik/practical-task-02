"""
qa/answer_generator.py
───────────────────────
Generates a structured answer from retrieved chunks using OpenAI GPT.
Returns the answer text + formatted source citations.
"""
from __future__ import annotations

import logging

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are an expert assistant on the NASA Systems Engineering Handbook (SP-2016-6105 Rev2).
Answer the user's question using ONLY the provided context chunks below.
Rules:
- Be precise and technical. Do not hallucinate content not in the context.
- Always cite sources using [Section X.Y, p.N] format.
- If the answer spans multiple sections, reference all relevant sections.
- If a figure or diagram is mentioned, say "See [Fig X]" and describe what it represents.
- If you cannot find the answer in the provided context, say so clearly.
- Expand acronyms on first use (e.g., "PDR (Preliminary Design Review)").
"""


def generate_answer(
    query: str,
    retrieved_hits: list[dict],
    max_context_chars: int = 12_000,
) -> dict:
    """
    Generate an answer to 'query' using 'retrieved_hits' as context.

    Returns:
        {
            "answer": str,
            "citations": list[dict],  # {section_id, section_title, pages}
            "has_figures": bool,
            "figures": list[dict],    # figure hits with paths
        }
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # ── Build context string ──────────────────────────────────────────────────
    context_parts: list[str] = []
    citations: list[dict] = []
    figures: list[dict] = []
    total_chars = 0

    for i, hit in enumerate(retrieved_hits):
        chunk_type = hit.get("chunk_type", "text")
        section_id = hit.get("section_id", "?")
        pages = hit.get("page_numbers", [])
        text = hit.get("text", "")

        if total_chars + len(text) > max_context_chars:
            break

        if chunk_type == "figure":
            fig_ref = hit.get("figure_ref", f"Fig {i+1}")
            ctx_block = f"[FIGURE: {fig_ref}]\nCaption: {hit.get('caption','')}\nPage: {pages[0] if pages else '?'}"
            figures.append(hit)
        elif chunk_type == "table":
            ctx_block = f"[TABLE: {hit.get('caption', f'Table {i+1}')}]\n{text}"
        else:
            ctx_block = text

        source_label = f"[Source {i+1}: Section {section_id}, p.{'/'.join(map(str, pages[:2]))}]"
        context_parts.append(f"{source_label}\n{ctx_block}")
        total_chars += len(ctx_block)

        citations.append(
            {
                "source_num": i + 1,
                "section_id": section_id,
                "section_title": hit.get("section_title", ""),
                "chapter_num": hit.get("chapter_num"),
                "chapter_title": hit.get("chapter_title", ""),
                "pages": pages,
                "chunk_type": chunk_type,
                "via_cross_ref": hit.get("_via_cross_ref", False),
            }
        )

    context_str = "\n\n---\n\n".join(context_parts)

    user_message = f"""Question: {query}

Context:
{context_str}

Please answer the question based on the context above."""

    # ── Call LLM ─────────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        answer = response.choices[0].message.content or "No answer generated."
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = f"Error generating answer: {e}"

    return {
        "answer": answer,
        "citations": citations,
        "has_figures": len(figures) > 0,
        "figures": figures,
    }
