"""Exportadores de informes a CSV y JSON Lines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = {
    "Departamento": "Departamento",
    "Provincia": "Provincia",
    "Distrito": "Distrito",
    "Modalidad de servicio": "ModalidadServicio",
    "Número de informe": "NumeroInforme",
    "Entidad": "Entidad",
    "Título del informe": "Descripcion",
    "Sector": "Sector",
    "Fecha de emisión": "FechaEmision",
    "Fecha de conclusión": "FechaFinEjecucion",
    "Fecha de publicación": "FechaPublicacion",
    "Personas con responsabilidad": "TotalFuncionarios",
    "Ficha resumen (URL)": "ResumenEjecutivo",
    "Informe (URL)": "ResumenInforme",
}


def export_reports(reports: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "informes_cusco.csv"
    jsonl_path = output_dir / "informes_cusco.jsonl"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS.keys())
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    label: report.get(source, "")
                    for label, source in CSV_FIELDS.items()
                }
            )

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for report in reports:
            jsonl_file.write(json.dumps(report, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path
