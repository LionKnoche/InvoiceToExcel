from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from datetime import date

try:
    from . import config_store
    from . import excel_store
    from . import format_utils
    from . import llm_client
    from . import ocr_service
    from . import mcp_excel_server
except ImportError:
    import config_store
    import excel_store
    import format_utils
    import llm_client
    import ocr_service
    import mcp_excel_server


@dataclass
class PipelineItem:
    file_path: Any
    ocr_text: str = ""
    selected_vendor: str = ""
    status: str = "Bereit"


def run_ocr(items: List[Dict[str, Any]], update_status, refresh_tree) -> None:
    """Run OCR + vendor classification for items in-place."""
    cfg = config_store.load_config()
    vendors_cfg = cfg.get("vendors", [])

    for item in items:
        item["status"] = "OCR laeuft..."
        refresh_tree()

        try:
            ocr_text = ocr_service.ocr_text(item["file_path"])
            item["ocr_text"] = ocr_text

            candidates = ocr_service.collect_regex_candidates(ocr_text, vendors_cfg)
            possible_vendors = candidates.get("possible_vendors", [])

            if possible_vendors:
                item["detected_vendor"] = possible_vendors[0]
                item["selected_vendor"] = possible_vendors[0]
            else:
                item["detected_vendor"] = "(Nicht erkannt)"
                item["selected_vendor"] = "(Unbekannt)"

            item["status"] = "Klassifiziert"
        except Exception as exc:
            item["status"] = f"Fehler: {str(exc)[:30]}"
            item["detected_vendor"] = "-"

        refresh_tree()

    update_status("OCR + Klassifikation abgeschlossen.")


def run_extraction(items: List[Dict[str, Any]], update_status, refresh_tree, log_fn=None) -> None:
    """Run OCR + LLM extraction for items in-place.

    update_status: function to update status label text
    refresh_tree: function to refresh UI tree
    """
    cfg = config_store.load_config()
    vendors_cfg = cfg.get("vendors", [])

    for item in items:
        item["status"] = "OCR laeuft..."
        refresh_tree()

        try:
            if not item.get("ocr_text"):
                ocr_text = ocr_service.ocr_text(item["file_path"])
                item["ocr_text"] = ocr_text

            item["status"] = "LLM laeuft..."
            refresh_tree()

            candidates = ocr_service.collect_regex_candidates(item["ocr_text"], vendors_cfg)

            selected = item.get("selected_vendor")
            if selected and not selected.startswith("("):
                candidates["possible_vendors"] = [selected]

            llm_result = llm_client.call_llm_for_invoice(
                ocr_text=item["ocr_text"],
                candidates=candidates,
                cfg=cfg,
                log_fn=log_fn,
            )

            if llm_result:
                if selected and not selected.startswith("("):
                    llm_result["vendor_name"] = selected

                inv_date_str = format_utils.to_iso_date(llm_result.get("invoice_date"))
                due_date_str = format_utils.to_iso_date(llm_result.get("due_date"))

                inv_date = date.fromisoformat(inv_date_str) if inv_date_str else date.today()
                due_date_obj = date.fromisoformat(due_date_str) if due_date_str else None

                raw_gross = llm_result.get("gross_amount")
                gross = format_utils.to_float(raw_gross) if raw_gross is not None else 0.0

                tax_rate = format_utils.to_float(llm_result.get("tax_rate", 0.19))
                if tax_rate > 1:
                    tax_rate = tax_rate / 100

                sheet_name = "Rechnungen"
                columns = [
                    "InvoiceId",
                    "InvoiceNumber",
                    "VendorName",
                    "InvoiceDate",
                    "DueDate",
                    "GrossAmount",
                    "TaxRate",
                    "Currency",
                    "Description",
                    "FilePath",
                ]

                for vendor in cfg.get("vendors", []):
                    if vendor.get("name") == selected:
                        category_name = vendor.get("category", "")
                        for cat in cfg.get("categories", []):
                            if cat.get("name") == category_name:
                                sheet_name = cat.get("sheet_name", sheet_name)
                                columns = cat.get("columns", columns)
                                break
                        break

                excel_data = {
                    "invoice_number": llm_result.get("invoice_number") or "UNKNOWN",
                    "vendor_name": llm_result.get("vendor_name") or selected or "Unknown",
                    "invoice_date": inv_date,
                    "due_date": due_date_obj,
                    "gross_amount": gross,
                    "tax_rate": tax_rate,
                    "currency": llm_result.get("currency", "EUR"),
                    "description": llm_result.get("description", ""),
                    "delivery_note": llm_result.get("delivery_note", ""),
                    "file_path": str(item["file_path"]),
                }

                try:
                    new_id = excel_store.append_invoice_with_schema(
                        data=excel_data,
                        sheet_name=sheet_name,
                        columns=columns,
                    )
                    item["status"] = f"OK (ID {new_id})"
                except Exception as exc:
                    item["status"] = f"Excel Fehler: {str(exc)[:30]}"
            else:
                item["status"] = "LLM leer"

        except Exception as exc:
            item["status"] = f"Fehler: {str(exc)[:30]}"

        refresh_tree()

    update_status("LLM-Extraktion abgeschlossen.")
