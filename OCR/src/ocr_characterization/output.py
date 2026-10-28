"""Escritura de JSON, CSV y Markdown de caracterizacion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schema import INFERRED_FIELDS, METADATA_FIELDS, SITUATION_FIELDS


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _citation(citations: dict, field: str) -> str:
    citation = citations.get(field)
    if not isinstance(citation, dict):
        return ""
    page = citation.get("pagina", "")
    text = citation.get("texto", "")
    return f"p. {page}: {text}" if page else str(text)


def write_outputs(
    source: Path,
    output_dir: Path,
    result: dict[str, Any],
    response: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    response_path = output_dir / f"{stem}.response.txt"

    payload = {
        "documento": source.name,
        "requiere_revision_humana": True,
        **result,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    response_path.write_text(response.rstrip() + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seccion", "situacion", "campo", "valor", "tipo_campo", "evidencia"),
        )
        writer.writeheader()
        for field in METADATA_FIELDS:
            writer.writerow(
                {
                    "seccion": "metadatos",
                    "situacion": "",
                    "campo": field,
                    "valor": _text(result["metadatos"].get(field)),
                    "tipo_campo": "documental",
                    "evidencia": _citation(result.get("citas_metadatos", {}), field),
                }
            )
        for situation in result["situaciones_adversas"]:
            citations = situation.get("citas", {})
            number = _text(situation.get("num_situacion_adversa"))
            for field in SITUATION_FIELDS:
                writer.writerow(
                    {
                        "seccion": "situacion_adversa",
                        "situacion": number,
                        "campo": field,
                        "valor": _text(situation.get(field)),
                        "tipo_campo": "inferido" if field in INFERRED_FIELDS else "documental",
                        "evidencia": _citation(citations, field),
                    }
                )

    lines = [f"# Caracterizacion: {source.name}", "", "## Metadatos", "", "| Campo | Valor | Evidencia |", "|---|---|---|"]
    for field in METADATA_FIELDS:
            value = _text(result["metadatos"].get(field)) or "No identificado"
            evidence = _citation(result.get("citas_metadatos", {}), field)
            lines.append(f"| `{field}` | {value} | {evidence} |")

    lines.extend(["", "## Situaciones adversas"])
    if not result["situaciones_adversas"]:
        lines.extend(["", "No se identificaron situaciones; revisar la respuesta OCR."])
    for situation in result["situaciones_adversas"]:
        number = _text(situation.get("num_situacion_adversa")) or "?"
        lines.extend(["", f"### Situacion {number}", "", "| Campo | Valor | Evidencia |", "|---|---|---|"])
        citations = situation.get("citas", {})
        for field in SITUATION_FIELDS:
            value = _text(situation.get(field)) or "No identificado"
            evidence = _citation(citations, field)
            if field in INFERRED_FIELDS:
                value += " *(propuesta inferida; validar)*"
            lines.append(f"| `{field}` | {value} | {evidence} |")

    if result["observaciones"]:
        lines.extend(["", "## Observaciones"])
        lines.extend(f"- {_text(item)}" for item in result["observaciones"])
    lines.extend(["", "> Borrador generado automaticamente. Validar valores y citas contra el PDF original."])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [response_path, json_path, csv_path, markdown_path]