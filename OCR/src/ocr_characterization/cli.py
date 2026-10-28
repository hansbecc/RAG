"""Interfaz de linea de comandos."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import argparse

from .characterize import characterize_markdown
from .easyocr_backend import extract_pdf_text as extract_easyocr_pdf_text
from .output import write_outputs
from .paddle_backend import extract_pdf_text
from .pytesseract_backend import (
    extract_pdf_text as extract_pytesseract_pdf_text,
    tesseract_language,
)
from .render import render_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Caracteriza informes PDF mediante motores OCR locales."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--files",
        nargs="+",
        default=["INFORME03.pdf", "INFORME04.pdf", "INFORME05.pdf"],
    )
    parser.add_argument(
        "--backend", choices=("paddle", "easyocr", "pytesseract"), default="paddle"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--lang", default="es")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida PDFs y parametros sin cargar PaddleOCR ni requerir GPU.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 72 <= args.dpi <= 400:
        print("Error: --dpi debe estar entre 72 y 400.", file=sys.stderr)
        return 2
    if args.backend == "pytesseract" and args.device != "cpu":
        print("Error: PyTesseract solo admite --device cpu.", file=sys.stderr)
        return 2
    pdf_paths = [args.input_dir / filename for filename in args.files]
    missing = [path for path in pdf_paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"No se encuentra el PDF: {path}", file=sys.stderr)
        return 2

    if args.dry_run:
        for path in pdf_paths:
            print(f"OK: {path}")
        print(f"Motor: {args.backend} (idioma={args.lang}, dispositivo={args.device})")
        print(f"DPI: {args.dpi}")
        print(f"Salida: {args.output_dir}")
        return 0

    if args.backend == "paddle" and sys.version_info >= (3, 13):
        print(
            "Error: PaddlePaddle requiere Python 3.10-3.12 en esta configuracion. "
            f"El interprete activo es {sys.version.split()[0]}; crea el entorno con Python 3.12.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.backend == "paddle":
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR no esta instalado. Instala paddleocr y paddlepaddle; "
                    "consulta README.md para CPU/GPU."
                ) from exc

            try:
                ocr = PaddleOCR(
                    lang=args.lang,
                    device=args.device,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except RuntimeError as exc:
                if "Engine 'paddle_static' is unavailable" not in str(exc):
                    raise
                raise RuntimeError(
                    "Falta el motor paddlepaddle en este entorno. Instala OCR con "
                    "python -m pip install -e '.[cpu]' --extra-index-url "
                    "https://www.paddlepaddle.org.cn/packages/stable/cpu/ "
                    "usando Python 3.10-3.12."
                ) from exc
            extract_text = extract_pdf_text
        elif args.backend == "easyocr":
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "EasyOCR no esta instalado. Instala el extra con "
                    "uv pip install --python .venv-easyocr/bin/python -e '.[easyocr]'."
                ) from exc

            ocr = easyocr.Reader([args.lang], gpu=args.device == "gpu")
            extract_text = extract_easyocr_pdf_text
        else:
            try:
                import pytesseract
            except ImportError as exc:
                raise RuntimeError(
                    "PyTesseract no esta instalado. Instala el extra con "
                    "uv pip install --python .venv-tesseract/bin/python -e '.[pytesseract]'."
                ) from exc

            ocr = pytesseract
            language = tesseract_language(args.lang)
            extract_text = lambda engine, paths: extract_pytesseract_pdf_text(
                engine, paths, language
            )

        for pdf_path in pdf_paths:
            print(f"Procesando {pdf_path}...")
            with tempfile.TemporaryDirectory(prefix="cgr_ocr_pages_") as temporary_dir:
                page_paths = render_pdf(pdf_path, Path(temporary_dir), dpi=args.dpi)
                response = extract_text(ocr, page_paths)
            result = characterize_markdown(response, pdf_path.name)
            outputs = write_outputs(pdf_path, args.output_dir, result, response)
            for output in outputs:
                print(f"  {output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())