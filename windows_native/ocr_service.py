from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import re
import sys


def _resolve_tesseract_path() -> Optional[Path]:
    env_path = os.getenv("TESSERACT_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", ""))
        candidate = meipass / "tesseract" / "tesseract.exe"
        if candidate.exists():
            return candidate

    default_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_path.exists():
        return default_path
    return None


def _resolve_poppler_path() -> Optional[Path]:
    env_path = os.getenv("POPPLER_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path if path.is_dir() else path.parent

    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", ""))
        candidate = meipass / "poppler"
        if candidate.exists():
            return candidate

    default_path = Path(r"C:\Program Files (x86)\poppler-25.12.0\Library\bin")
    if default_path.exists():
        return default_path
    return None

# OCR dependencies: pytesseract, pdf2image, Pillow
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image

    TESSERACT_PATH = _resolve_tesseract_path()
    POPPLER_PATH = _resolve_poppler_path()

    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)
        tessdata_dir = TESSERACT_PATH.parent / "tessdata"
        if tessdata_dir.exists():
            os.environ.setdefault("TESSDATA_PREFIX", str(tessdata_dir))
except Exception:
    pytesseract = None
    convert_from_path = None
    Image = None


def ocr_text(file_path: Path) -> str:
    """Simple OCR: supports PDF (via pdf2image) and images (via PIL)."""
    if pytesseract is None or convert_from_path is None or Image is None:
        raise RuntimeError("OCR libs fehlen: installiere pytesseract, pdf2image, pillow.")

    suffix = file_path.suffix.lower()
    texts: List[str] = []

    try:
        if suffix == ".pdf":
            poppler_path = POPPLER_PATH if POPPLER_PATH and POPPLER_PATH.exists() else None
            if poppler_path:
                pages = convert_from_path(str(file_path), poppler_path=str(poppler_path))
            else:
                pages = convert_from_path(str(file_path))
            for page in pages:
                texts.append(pytesseract.image_to_string(page, lang="deu+eng"))
        else:
            img = Image.open(file_path)
            texts.append(pytesseract.image_to_string(img, lang="deu+eng"))
    except Exception as exc:
        raise RuntimeError(f"OCR fehlgeschlagen: {exc}")

    return "\n".join(texts)


def collect_regex_candidates(
    text: str,
    vendors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Collects simple regex candidates for the LLM (amounts, percents, dates, vendors).
    """
    amounts = re.findall(r"\d+[.,]\d{2}", text)
    percents = re.findall(r"\d{1,2}\s*%", text)
    iso_dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    eu_dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", text)

    vendors_cfg = vendors or []
    possible_vendors: List[str] = []
    for vendor in vendors_cfg:
        name = vendor.get("name")
        patterns = vendor.get("patterns", [])
        if not name or not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if not pattern or not pattern.strip():
                continue
            # Treat patterns as literals, not regex.
            escaped = re.escape(pattern.strip())
            flexible_pattern = escaped.replace(r"\ ", r"\s+")
            try:
                if re.search(flexible_pattern, text, re.IGNORECASE):
                    possible_vendors.append(name)
                    break
            except re.error:
                if pattern.lower() in text.lower():
                    possible_vendors.append(name)
                    break

    due_keywords = [
        r"f\u00e4llig",
        r"faellig",
        r"zahlbar\s+bis",
        r"zahlungsziel",
        r"due\s+date",
        r"zahlbar\s+am",
        r"f\u00e4lligkeitsdatum",
        r"faelligkeitsdatum",
        r"zahlungstermin",
        r"zahlung bis",
    ]
    lines = text.splitlines()
    due_lines = []
    for line in lines:
        line_lower = line.lower()
        for keyword in due_keywords:
            if re.search(keyword, line_lower, re.IGNORECASE):
                due_lines.append(line.strip())
                break

    return {
        "possible_amounts": amounts,
        "possible_percents": percents,
        "possible_iso_dates": iso_dates,
        "possible_eu_dates": eu_dates,
        "possible_due_lines": due_lines,
        "possible_vendors": possible_vendors,
    }
