# InvoiceApp

InvoiceApp is a Windows desktop tool that OCRs invoices (PDF or image), extracts
structured fields with an LLM, and appends the results to Excel.

## Features
- GUI config editor with a human-in-the-loop pipeline
- Configurable categories, sheets, columns, and vendor-specific prompts
- LLM backends: local (Ollama) or cloud (Google Gemini via OpenAI-compatible API)
- Excel output with per-category schemas
- Programmatic processing via a simple Python API

## Project Layout
- `windows_native/config_editor.py` - Tkinter GUI for config and pipeline
- `windows_native/pipeline.py` - OCR + LLM extraction workflow
- `windows_native/ocr_service.py` - OCR helpers (Tesseract + pdf2image)
- `windows_native/llm_client.py` - OpenAI-compatible LLM client
- `windows_native/excel_store.py` - Excel read/write helpers
- `windows_native/config_store.py` - Config and data paths
- `windows_native/mcp_excel_server.py` - Programmatic processing entrypoints
- `build_exe.ps1`, `InvoiceApp.spec` - PyInstaller bundle

## Quick Start (Portable EXE)
1. Run `dist\InvoiceApp.exe` (no install required).
2. On first run it creates `%APPDATA%\InvoiceApp\invoice_config.json`.
3. Use the GUI to set provider, model, and API key.

The bundled EXE includes Tesseract and Poppler.

## Run From Source
Prerequisites:
- Python 3.12
- `pip install openpyxl requests pytesseract pdf2image pillow`
- Tesseract OCR and Poppler (for PDFs)

Run:
```
python windows_native\config_editor.py
```

## Configuration
Default config location:
`%APPDATA%\InvoiceApp\invoice_config.json`

Schema overview:
- `categories`: sheet definitions and column lists
- `vendors`: vendor name patterns and optional column prompts
- `llm`: provider, base URL, model, and API key

Example `llm` settings (Google Gemini via OpenAI-compatible API):
```
"llm": {
  "provider": "google",
  "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
  "api_key": "YOUR_API_KEY",
  "model": "gemini-2.5-flash"
}
```

## Environment Variables
Data paths:
- `INVOICE_DATA_DIR`
- `INVOICE_CONFIG_PATH`
- `INVOICE_EXCEL_PATH`
- `INVOICE_INCOMING_DIR`
- `INVOICE_LOG_PATH`

LLM:
- `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- Legacy aliases: `ULAMA_BASE_URL`, `ULAMA_API_KEY`, `ULAMA_MODEL`

OCR:
- `TESSERACT_PATH`
- `POPPLER_PATH`

## Workflow
1. Select invoice files in the GUI.
2. OCR runs and suggests a vendor.
3. Confirm or change vendor.
4. LLM extracts fields and writes to Excel.

Excel output:
`%APPDATA%\InvoiceApp\invoices.xlsx` (or overridden by env vars).

## Programmatic API
For automated runs, place files under the incoming folder and call:
```
from windows_native.mcp_excel_server import process_invoice_file
```

The incoming folder is `%APPDATA%\InvoiceApp\incoming` by default.

## Build
```
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```
Outputs:
- `dist\InvoiceApp.exe`
- `dist\InvoiceApp.zip`

The build script aborts if `windows_native\invoice_config.json` contains an API key.

## Security
API keys are stored locally in `%APPDATA%\InvoiceApp\invoice_config.json`.
Do not commit secrets to git.
