"""Renderizado temporal de paginas PDF para el modelo visual."""

from __future__ import annotations

from pathlib import Path


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Instala PyMuPDF para renderizar los documentos PDF.") from exc

    if dpi < 72 or dpi > 400:
        raise ValueError("DPI debe estar entre 72 y 400.")

    output_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    try:
        scale = dpi / 72
        matrix = pymupdf.Matrix(scale, scale)
        paths = []
        for page_number, page in enumerate(document, start=1):
            image_path = output_dir / f"page_{page_number:04d}.png"
            page.get_pixmap(matrix=matrix, alpha=False).save(image_path)
            paths.append(image_path)
        return paths
    finally:
        document.close()