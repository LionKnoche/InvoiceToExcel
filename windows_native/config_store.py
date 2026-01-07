from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional
import json
import os

APP_NAME = "InvoiceApp"


def get_app_data_dir() -> Path:
    env_dir = os.getenv("INVOICE_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    base_dir = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
    if base_dir:
        return Path(base_dir) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def get_config_path() -> Path:
    env_path = os.getenv("INVOICE_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return get_app_data_dir() / "invoice_config.json"


def get_excel_path() -> Path:
    env_path = os.getenv("INVOICE_EXCEL_PATH")
    if env_path:
        return Path(env_path)
    return get_app_data_dir() / "invoices.xlsx"


def get_incoming_dir() -> Path:
    env_path = os.getenv("INVOICE_INCOMING_DIR")
    if env_path:
        return Path(env_path)
    return get_app_data_dir() / "incoming"


def get_log_path() -> Path:
    env_path = os.getenv("INVOICE_LOG_PATH")
    if env_path:
        return Path(env_path)
    return get_app_data_dir() / "invoice_processing.log"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


TEMPLATE_CONFIG_PATH = Path(__file__).with_name("invoice_config.json")
CONFIG_PATH = get_config_path()

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_prompt": (
        "Nutze dein Wissen ueber typische deutsche Rechnungen. "
        "Wenn moeglich, erkenne den Lieferanten aus dem Text (z.B. Firmenname, IBAN, Steuernummer) "
        "und passe deine Interpretation der Felder daran an."
    ),
    "categories": [
        {
            "name": "Standard",
            "sheet_name": "Rechnungen",
            "columns": [
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
            ],
        }
    ],
    "vendors": [],
    "llm": {
        "provider": "ollama",
        "base_url": "http://localhost:11434/v1",
        "api_key": "",
        "model": "gemma3:4b",
    },
}


def _deepcopy_default() -> Dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_CONFIG))


def _normalize_categories(cfg: Dict[str, Any]) -> None:
    if "categories" not in cfg or not isinstance(cfg["categories"], list):
        cfg["categories"] = DEFAULT_CONFIG.get("categories", []).copy()
        return

    norm_categories = []
    for cat in cfg["categories"]:
        if not isinstance(cat, dict):
            continue
        name = cat.get("name") or "Standard"
        sheet_name = cat.get("sheet_name") or name
        columns = cat.get("columns")
        if not isinstance(columns, list) or not columns:
            columns = DEFAULT_CONFIG["categories"][0]["columns"].copy()
        norm_categories.append(
            {
                "name": name,
                "sheet_name": sheet_name,
                "columns": columns,
            }
        )
    cfg["categories"] = norm_categories


def _normalize_vendors(cfg: Dict[str, Any]) -> None:
    if "vendors" not in cfg or not isinstance(cfg["vendors"], list):
        cfg["vendors"] = []
        return

    norm_vendors = []
    for v in cfg["vendors"]:
        if not isinstance(v, dict):
            continue
        if "name" not in v or not isinstance(v["name"], str):
            continue
        if "patterns" not in v or not isinstance(v["patterns"], list):
            v["patterns"] = []
        category = v.get("category")
        if category is not None and not isinstance(category, str):
            category = None
        v["category"] = category
        norm_vendors.append(v)
    cfg["vendors"] = norm_vendors


def _normalize_llm(cfg: Dict[str, Any]) -> None:
    if "llm" not in cfg or not isinstance(cfg["llm"], dict):
        cfg["llm"] = DEFAULT_CONFIG["llm"].copy()
        return

    llm_cfg = cfg["llm"]
    for key, default_val in DEFAULT_CONFIG["llm"].items():
        if not isinstance(llm_cfg.get(key), str):
            llm_cfg[key] = default_val

    if llm_cfg.get("provider") == "google" and not llm_cfg.get("base_url"):
        llm_cfg["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai"


def _strip_secrets(cfg: Dict[str, Any]) -> None:
    llm_cfg = cfg.setdefault("llm", {})
    if "api_key" in llm_cfg:
        llm_cfg["api_key"] = ""


def _normalize_default_prompt(cfg: Dict[str, Any]) -> None:
    if "default_prompt" not in cfg or not isinstance(cfg["default_prompt"], str):
        cfg["default_prompt"] = DEFAULT_CONFIG["default_prompt"]


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _normalize_default_prompt(cfg)
    _normalize_categories(cfg)
    _normalize_vendors(cfg)
    _normalize_llm(cfg)
    return cfg


def load_config(
    path: Optional[Path] = None,
    on_error: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    target_path = path or get_config_path()

    if target_path.exists():
        try:
            with target_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            if on_error:
                on_error(f"Konfigurationsdatei konnte nicht gelesen werden: {exc}")
            cfg = _deepcopy_default()
    else:
        if TEMPLATE_CONFIG_PATH.exists():
            try:
                with TEMPLATE_CONFIG_PATH.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        if not cfg:
            cfg = _deepcopy_default()
        cfg = normalize_config(cfg)
        _strip_secrets(cfg)
        try:
            _ensure_parent_dir(target_path)
            target_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception:
            pass

    return normalize_config(cfg)


def save_config(cfg: Dict[str, Any], path: Optional[Path] = None) -> None:
    target_path = path or get_config_path()
    _ensure_parent_dir(target_path)
    target_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
