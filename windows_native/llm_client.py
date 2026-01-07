from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import json
import os
import re

import requests

LogFn = Callable[[str], None]


def _get_logger(log_fn: Optional[LogFn]) -> LogFn:
    if log_fn is None:
        return lambda _msg: None
    return log_fn


def _resolve_llm_settings(llm_cfg: Dict[str, Any]) -> Dict[str, str]:
    provider = os.getenv("LLM_PROVIDER") or llm_cfg.get("provider", "ollama")
    base_url = (
        os.getenv("ULAMA_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or llm_cfg.get("base_url")
        or "http://localhost:11434/v1"
    )
    api_key = (
        os.getenv("ULAMA_API_KEY")
        or os.getenv("LLM_API_KEY")
        or llm_cfg.get("api_key")
        or ""
    )
    model_name = (
        os.getenv("ULAMA_MODEL")
        or os.getenv("LLM_MODEL")
        or llm_cfg.get("model")
        or "gemma3:4b"
    )

    if provider == "google" and not base_url:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
    }


def call_llm_for_invoice(
    ocr_text: str,
    candidates: Dict[str, Any],
    cfg: Dict[str, Any],
    log_fn: Optional[LogFn] = None,
) -> Optional[Dict[str, Any]]:
    """
    Ruft ein LLM auf, um Rechnungsdaten strukturiert zu extrahieren.
    Erwartet, dass der Endpoint JSON zurueckgibt, das direkt in unser Schema passt.
    """
    log = _get_logger(log_fn)
    llm_cfg = cfg.get("llm", {})
    settings = _resolve_llm_settings(llm_cfg)
    provider = settings["provider"]
    base_url = settings["base_url"]
    api_key = settings["api_key"]
    model_name = settings["model_name"]

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        if provider == "google":
            headers["x-goog-api-key"] = api_key

    vendor_names = candidates.get("possible_vendors", []) or []
    vendor_prompt = ""
    selected_vendor_name = None
    column_prompts: Dict[str, Any] = {}
    selected_vendor: Optional[Dict[str, Any]] = None
    schema_columns: List[str] = []

    if vendor_names:
        selected_vendor_name = vendor_names[0]
        for vendor in cfg.get("vendors", []):
            if vendor.get("name") == selected_vendor_name:
                selected_vendor = vendor
                if isinstance(vendor.get("prompt"), str):
                    vendor_prompt = vendor["prompt"]

                raw_column_prompts = vendor.get("column_prompts", {}) or {}
                for col_name, col_config in raw_column_prompts.items():
                    if isinstance(col_config, str):
                        column_prompts[col_name] = col_config
                    elif isinstance(col_config, dict):
                        # Nur den Prompt-Text verwenden, keine Lookup-Konfiguration
                        column_prompts[col_name] = col_config.get("prompt", "")

                category_name = vendor.get("category", "")
                for cat in cfg.get("categories", []):
                    if cat.get("name") == category_name:
                        schema_columns = cat.get("columns", [])
                        break
                break

    if not vendor_prompt and isinstance(cfg.get("default_prompt"), str):
        vendor_prompt = cfg["default_prompt"]

    column_to_llm = {
        "InvoiceNumber": ("invoice_number", '"string oder null"'),
        "VendorName": ("vendor_name", '"string"'),
        "InvoiceDate": ("invoice_date", '"YYYY-MM-DD" (Rechnungsdatum)'),
        "DueDate": ("due_date", '"YYYY-MM-DD oder null" (Faelligkeitsdatum)'),
        "GrossAmount": ("gross_amount", "number (z.B. 123.45)"),
        "TaxRate": ("tax_rate", "number zwischen 0 und 1 (z.B. 0.19)"),
        "Currency": ("currency", '"EUR"'),
        "Description": ("description", '"string"'),
        "DeliveryNote": ("delivery_note", '"string oder null"'),
    }
    system_columns = {"InvoiceId", "FilePath"}

    def get_llm_field_name(column_name: str) -> str:
        if column_name in column_to_llm:
            return column_to_llm[column_name][0]
        return column_name.lower().replace(" ", "_")

    if schema_columns:
        llm_fields = []
        for col in schema_columns:
            if col in system_columns:
                continue
            if col in column_to_llm:
                field_name, field_type = column_to_llm[col]
                llm_fields.append(f'  "{field_name}": {field_type}')
            else:
                field_name = col.lower().replace(" ", "_")
                llm_fields.append(f'  "{field_name}": "string oder null"')

        json_structure = "{\n" + ",\n".join(llm_fields) + "\n}"
        log(f"[LLM] Schema-basierte Felder: {[col for col in schema_columns if col not in system_columns]}")
    else:
        json_structure = (
            "{\n"
            '  "invoice_number": "string oder null",\n'
            '  "invoice_date": "YYYY-MM-DD",\n'
            '  "gross_amount": number,\n'
            '  "description": "string"\n'
            "}"
        )
        log("[LLM] Fallback-Schema verwendet (keine Spalten definiert)")

    base_system_prompt = (
        "Du bist ein System zur Extraktion von Rechnungsdaten aus OCR-Text.\n"
        "Antworte AUSSCHLIESSLICH mit einem gueltigen JSON-Objekt, KEIN zusaetzlicher Text.\n"
        f"Das JSON muss GENAU diese Struktur haben (NUR diese Felder, keine anderen!):\n"
        f"{json_structure}\n\n"
        "WICHTIG:\n"
        "- Extrahiere NUR die oben genannten Felder!\n"
        "- due_date ist das FAELLIGKEITSDATUM, NICHT das Lieferdatum.\n"
        "- Nutze die regex_candidates als Hinweise.\n"
    )

    if selected_vendor_name:
        vendor_context = f"\nDer Lieferant (vendor_name) ist bereits festgelegt: {selected_vendor_name}.\n"
    else:
        vendor_context = "\nDer Lieferant ist unbekannt.\n"

    column_hints = ""
    if column_prompts:
        column_hints = "\nSpalten-spezifische Extraktionshinweise:\n"
        for col_name, col_prompt in column_prompts.items():
            if col_prompt:
                llm_field_name = get_llm_field_name(col_name)
                column_hints += f"- {llm_field_name}: {col_prompt}\n"

        log(f"[LLM] Vendor: {selected_vendor_name or 'Unbekannt'}")
        log(f"[LLM] Column Prompts geladen: {list(column_prompts.keys())}")
        log(f"[LLM] Column Hints:\n{column_hints}")

    system_prompt = base_system_prompt + vendor_context + vendor_prompt + column_hints

    log(f"[LLM] System-Prompt Laenge: {len(system_prompt)} Zeichen")
    if column_hints:
        log("[LLM] Column Hints enthalten: JA")
    else:
        log("[LLM] Column Hints enthalten: NEIN")

    due_lines = candidates.get("possible_due_lines", [])
    user_message = (
        "Extrahiere die Rechnungsdaten aus folgendem OCR-Text:\n\n"
        f"{ocr_text}\n\n"
        "Hinweise aus Regex-Suche:\n"
        f"- Moegliche Betraege: {candidates.get('possible_amounts', [])[:5]}\n"
        f"- Moegliche Steuersaetze: {candidates.get('possible_percents', [])[:3]}\n"
        f"- Moegliche Daten: {candidates.get('possible_iso_dates', []) + candidates.get('possible_eu_dates', [])[:3]}\n"
    )
    user_message = "".join(user_message)
    if due_lines:
        user_message += (
            f"- WICHTIG: Moegliche Faelligkeitszeilen (hier steht das due_date!): {due_lines[:3]}\n"
        )
    user_message += "\nGib NUR das JSON-Objekt zurueck, keine Erklaerung."

    if schema_columns and "DeliveryNote" in schema_columns:
        delivery_note_patterns = [
            r"\\b\\d{5}\\b",
            r"\\b\\d{4,6}\\b",
            r"Lieferschein[:\\s]+(\\d+)",
            r"LS[:\\s]+(\\d+)",
            r"Lieferschein-Nr[.:\\s]+(\\d+)",
        ]
        found_numbers = []
        for pattern in delivery_note_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            found_numbers.extend(matches[:3])

        if found_numbers:
            log(f"[LLM] Moegliche DeliveryNote-Nummern im OCR-Text gefunden: {found_numbers[:10]}")
        else:
            log("[LLM] Keine DeliveryNote-Nummern im OCR-Text gefunden (Regex-Suche)")
            lines_with_numbers = [
                line for line in ocr_text.splitlines() if re.search(r"\\d{4,}", line)
            ]
            if lines_with_numbers:
                log(f"[LLM] Zeilen mit Zahlen (moeglicherweise DeliveryNote): {lines_with_numbers[:5]}")

    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        content_clean = content.strip()
        json_start = content_clean.find("{")
        json_end = content_clean.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            content_clean = content_clean[json_start:json_end]

        result = json.loads(content_clean)

        if not isinstance(result, dict):
            log(f"[LLM] Antwort ist kein Dict: {type(result)}")
            return None

        if not result:
            log("[LLM] Leeres JSON zurueckgegeben")
            return None

        if selected_vendor_name:
            result["vendor_name"] = selected_vendor_name

        log("[LLM] Erfolgreich extrahiert:")
        for key, value in result.items():
            log(f"  - {key}: {value} (type: {type(value).__name__})")

        if "delivery_note" in result:
            log(f"[LLM] DeliveryNote gefunden: {result['delivery_note']}")
        else:
            log("[LLM] DeliveryNote NICHT gefunden in Ergebnis")
            log(f"[LLM] Verfuegbare Felder: {list(result.keys())}")

        return result

    except json.JSONDecodeError as exc:
        log(f"[LLM] JSON-Parse-Fehler: {exc}")
        if "content" in locals():
            log(f"[LLM] Antwort war: {content[:500]}")
        return None
    except KeyError as exc:
        log(f"[LLM] Fehlender Key in Antwort: {exc}")
        if "data" in locals():
            log(f"[LLM] Antwort-Struktur: {list(data.keys())}")
        return None
    except Exception as exc:
        log(f"[LLM] Fehler bei der Rechnungsextraktion: {type(exc).__name__}: {exc}")
        return None
