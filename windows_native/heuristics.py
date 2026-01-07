from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import re

try:
    from . import ocr_service
except ImportError:
    import ocr_service

LlmFn = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]


def extract_fields_stub(
    file_path: Path,
    vendors_cfg: List[Dict[str, Any]],
    llm_fn: Optional[LlmFn] = None,
) -> Dict[str, Any]:
    """
    Simple OCR + regex stub with optional LLM extraction.
    """
    text = ocr_service.ocr_text(file_path)

    candidates = ocr_service.collect_regex_candidates(text, vendors_cfg)

    if llm_fn is not None:
        llm_result = llm_fn(text, candidates)
        if isinstance(llm_result, dict):
            return llm_result

    invoice_number = re.search(
        r"(Rechnungs?-?nr\\.?|Invoice\\s*No\\.?)\\s*[:\\-]?\\s*([A-Za-z0-9\\-_/]+)",
        text,
        re.IGNORECASE,
    )
    invoice_number = invoice_number.group(2) if invoice_number else "UNKNOWN"

    iso_date = re.search(r"(\\d{4}-\\d{2}-\\d{2})", text)
    eu_date = re.search(r"(\\d{2}\\.\\d{2}\\.\\d{4})", text)
    invoice_date = None
    if iso_date:
        invoice_date = iso_date.group(1)
    elif eu_date:
        d, mth, y = eu_date.group(1).split(".")
        invoice_date = f"{y}-{mth}-{d}"
    else:
        invoice_date = date.today().isoformat()

    due_date = None

    amt_cur = re.search(r"(\\d+[.,]\\d{2})\\s*(EUR|USD|GBP|CHF)", text, re.IGNORECASE)
    currency = "EUR"
    gross_amount = 0.0
    if amt_cur:
        gross_amount = float(amt_cur.group(1).replace(",", "."))
        currency = amt_cur.group(2).upper()

    tax_rate_match = re.search(r"(\\d{1,2})\\s*%", text)
    tax_rate = float(tax_rate_match.group(1)) / 100 if tax_rate_match else 0.19

    possible_vendors = candidates.get("possible_vendors", []) or []
    if possible_vendors:
        vendor_name = possible_vendors[0]
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        vendor_name = lines[0][:80] if lines else "Unknown Vendor"

    description = "Auto-extracted (regex stub without LLM)"

    return {
        "invoice_number": invoice_number,
        "vendor_name": vendor_name,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "gross_amount": gross_amount,
        "tax_rate": tax_rate,
        "currency": currency,
        "description": description,
    }
