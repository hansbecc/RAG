"""Adaptador de PaddleOCR 3.x para extraer texto de paginas renderizadas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _recognized_text(result: Any) -> str:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("PaddleOCR devolvio un resultado sin estructura JSON.")

    data = payload.get("res", payload)
    texts = data.get("rec_texts", []) if isinstance(data, dict) else []
    return "\n".join(text.strip() for text in texts if isinstance(text, str) and text.strip())


def extract_pdf_text(ocr: Any, page_paths: list[Path]) -> str:
    pages = []
    for page_number, image_path in enumerate(page_paths, start=1):
        page_results = ocr.predict(str(image_path))
        page_text = "\n".join(
            text for text in (_recognized_text(item) for item in page_results) if text
        )
        pages.append(f"[OCR_PAGE:{page_number}]\n\n{page_text}")
    return "\n\n".join(pages)