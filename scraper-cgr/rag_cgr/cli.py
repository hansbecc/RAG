"""Comandos CLI para indexar PDFs y consultar reportes analíticos."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from rag_cgr.pipeline import answer_question, ingest_pdf_directory
from rag_cgr.settings import RAGSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline RAG para PDFs de la Contraloría.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Procesa e indexa archivos PDF en Qdrant.")
    ingest.add_argument("--pdf-dir", type=Path, default=None)
    ask = commands.add_parser("ask", help="Genera un reporte analítico con citas.")
    ask.add_argument("question", help="Pregunta en lenguaje natural.")
    parser.add_argument("--verbose", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = RAGSettings.from_env()
    if args.command == "ingest":
        page_count = await ingest_pdf_directory(args.pdf_dir, settings)
        print(f"Ingesta terminada: {page_count} páginas procesadas.")
        return 0
    print(await answer_question(args.question, settings))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
