"""Interfaz de línea de comandos del scraper."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

from scraper_cgr.client import ContraloriaClient, ScraperError
from scraper_cgr.export import export_reports
from scraper_cgr.filters import DEFAULT_KEYWORDS, matches_keywords


def _report_id(report: dict[str, Any]) -> str:
    return str(report.get("CodigoInforme") or report.get("NumeroInforme") or "")


def _load_checkpoint(path: Path, signature: dict[str, Any], fresh: bool) -> tuple[int, list[dict[str, Any]]]:
    if fresh or not path.exists():
        return 1, []

    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("filters") != signature:
        raise ScraperError(
            f"El checkpoint {path} se creó con otros filtros. Usa --fresh para comenzar de nuevo."
        )
    return int(checkpoint.get("next_page", 1)), checkpoint.get("reports", [])


def _save_checkpoint(
    path: Path,
    signature: dict[str, Any],
    next_page: int,
    reports: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            {"filters": signature, "next_page": next_page, "reports": reports},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _download_pdfs(reports: list[dict[str, Any]], output_dir: Path) -> int:
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; CGRPublicDataResearch/1.0)"}
        )
        for report in reports:
            report_id = _report_id(report) or "informe"
            safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", report_id)
            for kind, field in (("resumen", "ResumenEjecutivo"), ("informe", "ResumenInforme")):
                url = report.get(field)
                destination = pdf_dir / f"{safe_id}_{kind}.pdf"
                if not url or destination.exists():
                    continue
                response = session.get(url, timeout=60, stream=True)
                response.raise_for_status()
                first_chunk = next(response.iter_content(chunk_size=8192), b"")
                if not first_chunk.startswith(b"%PDF"):
                    response.close()
                    continue
                with destination.open("wb") as pdf_file:
                    pdf_file.write(first_chunk)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            pdf_file.write(chunk)
                response.close()
                downloaded += 1

    return downloaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recolecta informes de control sobre obras y proyectos en Cusco."
    )
    parser.add_argument("--department", default="CUSCO", help="Región/departamento (por defecto: CUSCO).")
    parser.add_argument("--limit", type=int, default=3000, help="Máximo de informes coincidentes.")
    parser.add_argument("--page-size", type=int, default=1000, help="Filas por consulta (máximo observado: 1000).")
    parser.add_argument("--delay", type=float, default=0.4, help="Pausa entre páginas, en segundos.")
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="Términos separados por comas, buscados en el título del informe.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directorio de resultados.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/checkpoint.json"),
        help="Punto de reanudación guardado después de cada página.",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignora el checkpoint y empieza desde la página 1.")
    parser.add_argument("--download-pdfs", action="store_true", help="Descarga ficha resumen e informe para cada coincidencia.")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise ScraperError("--limit debe ser mayor que cero.")
    if not 1 <= args.page_size <= 1000:
        raise ScraperError("--page-size debe estar entre 1 y 1000.")

    keywords = tuple(word.strip() for word in args.keywords.split(",") if word.strip())
    signature = {
        "department": args.department.upper(),
        "keywords": list(keywords),
        "page_size": args.page_size,
    }
    start_page, reports = _load_checkpoint(args.checkpoint, signature, args.fresh)
    known_ids = {_report_id(report) for report in reports if _report_id(report)}
    client = ContraloriaClient(args.department, delay_seconds=args.delay)
    page_number = start_page

    while len(reports) < args.limit:
        page = client.fetch_page(page_number, args.page_size)
        if not page:
            break

        added = 0
        for report in page:
            if not matches_keywords(report, keywords):
                continue
            report_id = _report_id(report)
            if report_id and report_id in known_ids:
                continue
            reports.append(report)
            if report_id:
                known_ids.add(report_id)
            added += 1
            if len(reports) >= args.limit:
                break

        next_page = page_number + 1
        _save_checkpoint(args.checkpoint, signature, next_page, reports)
        print(
            f"Página {page_number}: {added} coincidencias nuevas; "
            f"{len(reports)}/{args.limit} acumuladas.",
            flush=True,
        )
        if len(page) < args.page_size:
            break
        page_number = next_page

    csv_path, jsonl_path = export_reports(reports[: args.limit], args.output_dir)
    print(f"Exportados {min(len(reports), args.limit)} informes:")
    print(f"  CSV:   {csv_path}")
    print(f"  JSONL: {jsonl_path}")

    if args.download_pdfs:
        print(f"PDF descargados: {_download_pdfs(reports[: args.limit], args.output_dir)}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (ScraperError, requests.RequestException, OSError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
