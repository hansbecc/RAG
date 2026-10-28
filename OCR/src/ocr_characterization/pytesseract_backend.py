"""Adaptador de PyTesseract para paginas PDF renderizadas como imagen."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_LANGUAGE_CODES = {
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "pt": "por",
}


def tesseract_language(language: str) -> str:
    return "+".join(_LANGUAGE_CODES.get(code, code) for code in language.split("+"))


def extract_pdf_text(ocr: Any, page_paths: list[Path], language: str = "spa") -> str:
    pages = []
    for page_number, image_path in enumerate(page_paths, start=1):
        page_text = ocr.image_to_string(str(image_path), lang=language).strip()
        pages.append(f"[OCR_PAGE:{page_number}]\n\n{page_text}")
    return "\n\n".join(pages)