"""Adaptador de EasyOCR para extraer texto de paginas renderizadas."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_pdf_text(ocr: Any, page_paths: list[Path]) -> str:
    pages = []
    for page_number, image_path in enumerate(page_paths, start=1):
        recognized = ocr.readtext(str(image_path), detail=0, paragraph=False)
        page_text = "\n".join(
            text.strip()
            for text in recognized
            if isinstance(text, str) and text.strip()
        )
        pages.append(f"[OCR_PAGE:{page_number}]\n\n{page_text}")
    return "\n\n".join(pages)