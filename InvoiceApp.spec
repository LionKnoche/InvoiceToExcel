# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

project_dir = os.path.abspath(os.getcwd())
entry_script = os.path.join(project_dir, "windows_native", "config_editor.py")

def _collect_tree(root, prefix):
    entries = []
    for dirpath, _, filenames in os.walk(root):
        rel_path = os.path.relpath(dirpath, root)
        target_dir = prefix if rel_path == "." else os.path.join(prefix, rel_path)
        for filename in filenames:
            entries.append((os.path.join(dirpath, filename), target_dir))
    return entries

datas = [
    (os.path.join(project_dir, "windows_native", "invoice_config.json"), "windows_native"),
]

tesseract_exe = os.getenv("TESSERACT_PATH") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tesseract_root = os.path.dirname(tesseract_exe) if tesseract_exe and os.path.exists(tesseract_exe) else None
if tesseract_root:
    datas += _collect_tree(tesseract_root, "tesseract")

poppler_path = os.getenv("POPPLER_PATH") or r"C:\Program Files (x86)\poppler-25.12.0\Library\bin"
if poppler_path and os.path.isfile(poppler_path):
    poppler_path = os.path.dirname(poppler_path)
if poppler_path and os.path.exists(poppler_path):
    datas += _collect_tree(poppler_path, "poppler")

icon_path = os.path.join(project_dir, "assets", "app.ico")
icon = icon_path if os.path.exists(icon_path) else None

a = Analysis(
    [entry_script],
    pathex=[project_dir, os.path.join(project_dir, "windows_native")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "pipeline",
        "config_store",
        "ocr_service",
        "llm_client",
        "excel_store",
        "excel_tools",
        "format_utils",
        "heuristics",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries, 
    a.zipfiles,
    a.datas,
    [],
    name="InvoiceApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=icon,
)
