"""Extraccion conservadora de campos desde texto OCR."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schema import METADATA_FIELDS, SITUATION_FIELDS, empty_result


SITUATION_HEADING = re.compile(
    r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:situaci[oó]n\s+adversa|situaci[oó]n\s+de\s+riesgo)"
    r"\s*(?:(?:n[úu]m(?:ero)?\.?\s*)|(?:n\.?\s*[°º]\s*))?[°º#.:\-]?\s*(\d+)\b[^\n]*"
)
REPORT_ID = re.compile(
    r"(?i)\b(?:informe(?:\s+de\s+[\wáéíóúñ ]+)?|oficio)\s*(?:n[úu]m(?:ero)?\.?\s*)?"
    r"n?\.?[°º#.\-]?\s*([0-9]{1,4}-\d{4}-[A-Z0-9/.-]+)"
)
DATE = re.compile(
    r"(?i)\b(?:\d{1,2}\s+de\s+[a-záéíóú]+\s+de\s+)?(20\d{2})\b"
)
MONEY = re.compile(r"(?i)(?:S/\.?\s*|PEN\s*)([\d,.]+)")
PAGE_MARKER = re.compile(r"\[OCR_PAGE:(\d+)\]")
FIELD_MARKERS = {
    "condicion_texto": re.compile(r"(?i)^\s*(?:condici[oó]n|situaci[oó]n advertida)\s*[:\-]"),
    "criterio_normativo_texto": re.compile(r"(?i)^\s*(?:criterio|normativa aplicable|base legal)\s*[:\-]"),
    "riesgo_efecto_texto": re.compile(r"(?i)^\s*(?:consecuencia|efecto|riesgo)\s*[:\-]"),
    "evidencia_soporte": re.compile(r"(?i)^\s*(?:evidencia|documentaci[oó]n revisada|fuente)\s*[:\-]"),
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n#:-")


def _first_match(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _page_for_value(text: str, value: str, start: int = 0) -> int | None:
    position = text.find(value, start)
    if position < 0:
        return None
    markers = list(PAGE_MARKER.finditer(text, 0, position))
    return int(markers[-1].group(1)) if markers else None


def _paragraphs(text: str) -> list[str]:
    return [_clean(part) for part in re.split(r"\n\s*\n", text) if _clean(part)]


def characterize_markdown(markdown: str, source_name: str) -> dict[str, Any]:
    result = empty_result()
    metadata = result["metadatos"]
    report_id = _first_match(REPORT_ID, markdown)
    if report_id:
        metadata["id_informe"] = report_id
        result.setdefault("citas_metadatos", {})["id_informe"] = {
            "pagina": _page_for_value(markdown, report_id),
            "texto": report_id,
        }

    year = _first_match(DATE, markdown)
    if year:
        metadata["anio_emision"] = year
        result.setdefault("citas_metadatos", {})["anio_emision"] = {
            "pagina": _page_for_value(markdown, year),
            "texto": year,
        }

    money = _first_match(MONEY, markdown)
    if money:
        metadata["monto_inversion_soles"] = f"S/ {money}"
        result.setdefault("citas_metadatos", {})["monto_inversion_soles"] = {
            "pagina": _page_for_value(markdown, money),
            "texto": f"S/ {money}",
        }

    matches = list(SITUATION_HEADING.finditer(markdown))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        section = markdown[match.end():end].strip()
        paragraphs = _paragraphs(section)
        title = _clean(match.group(0))
        title_page = _page_for_value(markdown, title, match.start())
        situation: dict[str, Any] = {field: None for field in SITUATION_FIELDS}
        situation["num_situacion_adversa"] = int(match.group(1))
        situation["titulo_situacion_adversa"] = title
        situation["citas"] = {
            "titulo_situacion_adversa": {"pagina": title_page, "texto": title}
        }

        for field, marker in FIELD_MARKERS.items():
            for paragraph in paragraphs:
                lines = paragraph.splitlines()
                if marker.match(lines[0]):
                    extracted = _clean(" ".join([marker.sub("", lines[0], count=1), *lines[1:]]))
                    situation[field] = extracted or None
                    if extracted:
                        situation["citas"][field] = {
                            "pagina": _page_for_value(markdown, extracted, match.end()),
                            "texto": extracted[:500],
                        }
                    break

        if situation["condicion_texto"] is None and paragraphs:
            candidate = next(
                (paragraph for paragraph in paragraphs if len(paragraph) > 80),
                None,
            )
            if candidate:
                situation["condicion_texto"] = candidate
                situation["citas"]["condicion_texto"] = {
                    "pagina": _page_for_value(markdown, candidate, match.end()),
                    "texto": candidate[:500],
                }

        if situation["criterio_normativo_texto"] is None:
            norms = [
                paragraph for paragraph in paragraphs
                if re.search(r"(?i)\b(?:RNE|NTP|Ley|Decreto|Directiva|Reglamento|Resoluci[oó]n)\b", paragraph)
            ]
            if norms:
                situation["criterio_normativo_texto"] = " ".join(norms)
                situation["citas"]["criterio_normativo_texto"] = {
                    "pagina": _page_for_value(markdown, " ".join(norms), match.end()),
                    "texto": " ".join(norms)[:500],
                }

        result["situaciones_adversas"].append(situation)

    result["observaciones"].append(
        f"Caracterizacion heuristica desde OCR de PaddleOCR: {source_name}. "
        "Los campos no identificados o que requieren interpretacion semantica se dejan vacios; "
        "validar texto, paginas y clasificaciones contra el PDF original."
    )
    return result