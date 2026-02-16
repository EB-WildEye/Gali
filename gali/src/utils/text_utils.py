"""
Gali - Text Utilities
======================
Helper functions for text cleaning, normalisation, and
filename-based metadata extraction.

These utilities are consumed primarily by the ``IngestionPipeline``
and should remain stateless and side-effect-free.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


# ── Non-printable character pattern ────────────────────────────────────
# Matches control characters (C0/C1), except \n, \r, \t which we handle
# separately. Also catches BOM, zero-width chars, soft hyphens, etc.
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufeff\u200b\u200c\u200d\u200e\u200f\u00ad\u2060\ufffe]")


# ── Public API ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Sanitise raw document text for embedding.

    Steps:
        1. Unicode NFC normalisation (canonical composition) –
           ensures consistent representation of Hebrew characters
           and their diacritical marks (nikkud).
        2. Strip non-printable / zero-width characters and formatting
           artifacts (BOM, soft hyphens, directional marks).
        3. Collapse runs of horizontal whitespace (spaces, tabs,
           non-breaking spaces) into a single space, *preserving*
           newlines.
        4. Strip leading / trailing whitespace from every line.
        5. Collapse 3+ consecutive blank lines to 2.

    Args:
        text: Raw text extracted from a source file.

    Returns:
        Cleaned, normalised text ready for chunking.
    """
    text = unicodedata.normalize("NFC", text)
    text = _NON_PRINTABLE_RE.sub("", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Protocol-type keywords → department mapping ───────────────────────
# Case-insensitive keyword matching is used against the full filename.
# Extend this map as new source documents are added to the knowledge base.
_PROTOCOL_KEYWORDS: list[tuple[str, str]] = [
    ("induced", "gynecology"),
    ("missed", "gynecology"),
]

_DEFAULT_PROTOCOL_TYPE = "unknown"
_DEFAULT_DEPARTMENT = "general"


def extract_metadata_from_filename(filename: str) -> dict[str, str]:
    """
    Derive structured metadata from a document filename using
    case-insensitive fuzzy keyword matching.

    Scans the full filename (stem) for known protocol-type keywords
    (``"induced"``, ``"missed"``, etc.) regardless of position or
    surrounding delimiters.

    Examples::

        "induced_protocol.txt"     → protocol_type="induced", department="gynecology"
        "Missed_Abortion_v2.pdf"   → protocol_type="missed",  department="gynecology"
        "INDUCED-protocol.docx"    → protocol_type="induced", department="gynecology"
        "random_document.txt"      → protocol_type="unknown", department="general"

    Args:
        filename: The file's name (stem + extension), **not** the full path.

    Returns:
        dict with keys ``protocol_type`` and ``department``.
    """
    stem = Path(filename).stem.lower()

    for keyword, department in _PROTOCOL_KEYWORDS:
        if keyword in stem:
            return {"protocol_type": keyword, "department": department}

    return {"protocol_type": _DEFAULT_PROTOCOL_TYPE, "department": _DEFAULT_DEPARTMENT}
