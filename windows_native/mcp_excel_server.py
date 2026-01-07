from datetime import date
from pathlib import Path
from typing import Optional, List, Dict, Any
import re
import sys
from datetime import datetime

# Import excel_store (funktioniert sowohl als Package als auch standalone)
try:
    from . import excel_store
    from . import config_store
    from . import excel_tools
    from . import heuristics
    from . import llm_client
    from . import ocr_service
except ImportError:
    # Fallback für direkte Ausführung
    _module_dir = Path(__file__).parent
    if str(_module_dir) not in sys.path:
        sys.path.insert(0, str(_module_dir))
    import excel_store
    import config_store
    import excel_tools
    import heuristics
    import llm_client
    import ocr_service

# Eingangsordner fuer Rechnungen
INCOMING_DIR = config_store.get_incoming_dir()
INCOMING_DIR.mkdir(parents=True, exist_ok=True)

# Log-Datei fuer Debug-Ausgaben
LOG_PATH = config_store.get_log_path()

def _log(message: str) -> None:
    """Schreibt Log-Nachricht sowohl in Konsole als auch in Log-Datei."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    # In Konsole (falls Terminal offen)
    print(log_message)
    
    # In Log-Datei
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception:
        pass  # Fehler beim Schreiben ignorieren

def reset_config_cache() -> None:
    """Setzt den Config-Cache zurück, damit die Config neu geladen wird."""
    return


def get_invoice_config() -> Dict[str, Any]:
    """
    Laedt die Konfiguration aus invoice_config.json (oder legt eine Default-Datei an).

    Struktur:
    {
        "default_prompt": "Text fuer das LLM",
        "vendors": [
            {
                "name": "Rinke",
                "patterns": ["rinke", "rinke\\s+gmbh"],
                "prompt": "Optionaler vendor-spezifischer Prompt"
            },
            ...
        ]
    }
    """
    return config_store.load_config()


def get_category_by_name(cfg: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Sucht eine Kategorie nach Name in der Konfiguration."""
    for cat in cfg.get("categories", []):
        if cat.get("name") == name:
            return cat
    return None


def get_vendor_category(cfg: Dict[str, Any], vendor_name: str) -> Optional[Dict[str, Any]]:
    """
    Liefert die Kategorie-Konfiguration fuer einen Vendor-Namen,
    basierend auf dem category-Feld des Vendor-Eintrags.
    """
    for v in cfg.get("vendors", []):
        if v.get("name") == vendor_name:
            cat_name = v.get("category")
            if cat_name:
                return get_category_by_name(cfg, cat_name)
            break
    return None

def append_invoice(
    invoice_number: str,
    vendor_name: str,
    invoice_date: str,
    due_date: Optional[str],
    gross_amount: float,
    tax_rate: float,
    currency: str,
    description: str,
    file_path: str,
) -> Dict[str, Any]:
    """
    Fügt eine Rechnung in Excel ein.
    Datum-Format: YYYY-MM-DD
    """
    try:
        inv_date = date.fromisoformat(invoice_date)
        due = date.fromisoformat(due_date) if due_date else None #
    except ValueError as e: # valueerror is an exception that is raised when a value is not valid
        return {"ok": False, "error": f"Ungültiges Datumsformat: {e}"} # literal means that the string is not interpolated; f means that the string is formatted

    new_id = excel_store.append_invoice(
        invoice_number=invoice_number,
        vendor_name=vendor_name,
        invoice_date=inv_date,
        due_date=due,
        gross_amount=gross_amount,
        tax_rate=tax_rate,
        currency=currency,
        description=description,
        file_path=file_path,
    )
    return {"ok": True, "invoiceId": new_id}


def read_invoices() -> List[Dict[str, Any]]:
    """
    Gibt alle Rechnungen aus der Excel-Tabelle zurück.
    """
    return excel_store.read_invoices()


def _ocr_text(file_path: Path) -> str:
    return ocr_service.ocr_text(file_path)


def _collect_regex_candidates(text: str) -> Dict[str, Any]:
    cfg = get_invoice_config()
    vendors_cfg = cfg.get("vendors", [])
    return ocr_service.collect_regex_candidates(text, vendors_cfg)


def _call_llm_for_invoice(ocr_text: str, candidates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ruft ein LLM (OpenAI-kompatible API) auf, um Rechnungsdaten strukturiert zu extrahieren.
    """
    cfg = get_invoice_config()
    return llm_client.call_llm_for_invoice(
        ocr_text=ocr_text,
        candidates=candidates,
        cfg=cfg,
        log_fn=_log,
    )


def _extract_fields_stub(file_path: Path) -> Dict[str, Any]:
    """
    Einfacher OCR+Regex-Stub.
    TODO: Ersetze Regex durch LLM-Extraktion (z.B. Ulama) fuer robuste Ergebnisse.

    Die Vendor-Erkennung und LLM-Prompts sind konfigurierbar:
    - Konfiguration: invoice_config.json (gleicher Ordner wie dieses Modul)
    - Bearbeitung per GUI: config_editor.py starten (Python-Script im Ordner windows_native)
    """
    cfg = get_invoice_config()
    vendors_cfg = cfg.get("vendors", [])
    return heuristics.extract_fields_stub(
        file_path=file_path,
        vendors_cfg=vendors_cfg,
        llm_fn=_call_llm_for_invoice,
    )


def process_invoice_file(file_path: str) -> Dict[str, Any]:
    """
    Liest eine Rechnungsdatei, extrahiert Felder (Stub) und schreibt sie nach Excel.
    Erwartet: invoice_number, vendor_name, invoice_date (YYYY-MM-DD), due_date (YYYY-MM-DD|None),
              gross_amount, tax_rate, currency, description
    """
    path = Path(file_path)

    # Pfad muss im definierten Eingangsordner liegen
    try:
        path.relative_to(INCOMING_DIR)
    except ValueError:
        return {"ok": False, "error": f"File must be under {INCOMING_DIR}"}

    if not path.exists():
        return {"ok": False, "error": f"File not found: {path}"}

    try:
        data = _extract_fields_stub(path)
    except NotImplementedError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Extract failed: {e}"}

    def _to_iso_date(value: Optional[str]) -> Optional[str]:
        """Konvertiert verschiedene Datumsformate nach YYYY-MM-DD."""
        if not value:
            return None
        v = value.strip()
        # Bereits ISO-Format
        try:
            date.fromisoformat(v)
            return v
        except Exception:
            pass
        # Versuche DD.MM.YYYY
        m = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", v)
        if m:
            d, mth, y = m.groups()
            return f"{y}-{mth}-{d}"
        # Fallback: None, löst weiter unten Fehler aus
        return None

    def _to_float(value: Any) -> float:
        """Konvertiert verschiedene Formate zu float (z.B. "19.00%" -> 0.19)."""
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        # Entferne Prozentzeichen und konvertiere zu Dezimalzahl (0-1)
        if "%" in s:
            s = s.replace("%", "").strip()
            num = float(s)
            return num / 100.0  # 19% -> 0.19
        return float(s)

    try:
        inv_iso = _to_iso_date(data.get("invoice_date"))
        due_iso = _to_iso_date(data.get("due_date"))
        if not inv_iso:
            raise ValueError(f"Unsupported invoice_date format: {data.get('invoice_date')}")

        inv_date = date.fromisoformat(inv_iso)
        due = date.fromisoformat(due_iso) if due_iso else None
    except Exception as e:
        return {"ok": False, "error": f"Date parse error: {e}"}

    try:
        new_id = excel_store.append_invoice(
            invoice_number=data["invoice_number"],
            vendor_name=data["vendor_name"],
            invoice_date=inv_date,
            due_date=due,
            gross_amount=_to_float(data["gross_amount"]),
            tax_rate=_to_float(data["tax_rate"]),
            currency=data["currency"],
            description=data.get("description", ""),
            file_path=str(path),
        )
    except Exception as e:
        return {"ok": False, "error": f"Excel write failed: {e}"}

    return {"ok": True, "invoiceId": new_id, "fields": data}


if __name__ == "__main__":
    # Hinweis: Konfiguration der Vendor-Regexe und LLM-Prompts erfolgt über config_editor.py.
    # Dieses Modul wird normalerweise über den MCP-Server verwendet.
    pass
