from __future__ import annotations

from datetime import date
from typing import Any, Optional
import re


def to_iso_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = str(value).strip()
    try:
        date.fromisoformat(v)
        return v
    except Exception:
        pass
    m = re.match(r"^(\\d{2})\\.(\\d{2})\\.(\\d{4})$", v)
    if m:
        d, mth, y = m.groups()
        return f"{y}-{mth}-{d}"
    return None


def to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return 0.0
    s = str(value).strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    s = s.replace("%", "").replace("EUR", "").strip()
    try:
        return float(s)
    except Exception:
        return 0.0
