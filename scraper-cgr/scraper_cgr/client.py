"""Cliente del endpoint público que utiliza el buscador de la Contraloría."""

from __future__ import annotations

import json
import time
from typing import Any

import requests


SEARCH_URL = (
    "https://buscadorinformes.contraloria.gob.pe/BuscadorCGR/Informes/"
    "BusquedaInformesCGR.ashx"
)


class ScraperError(RuntimeError):
    """Error recuperable o respuesta inesperada del buscador."""


def build_params(department: str, page_number: int, page_size: int) -> dict[str, str]:
    """Construye los parámetros observados en las peticiones del sitio oficial."""
    return {
        "ActionPage": "TransportType",
        "Action": "loadInformesElastic",
        "PageSize": str(page_size),
        "PageNumber": str(page_number),
        "pGeneral": "",
        "pAnio": "",
        "pFechaInicio": "01/01/1800",
        "pFechaHasta": "01/01/2500",
        "pEntidad": "",
        "pSector": "",
        "pModalidadServicio": ' NOT "REPORTE DE AVANCE"',
        "pReconstruccion": "",
        "pReconstruccionDepa": "",
        "pServicio": "",
        "pDepartamento": json.dumps(department.upper(), ensure_ascii=False),
        "pResponsabilidad": "",
        "pCovid": "",
        "pCovidDepa": "",
        "pNivelGobierno": "",
        "pProvincia": "",
        "pDistrito": "",
        "pAnioFinEjec": "",
        "pEsConResponsabilidad": "",
        "pEvento": "",
        "pOperativo": "",
        "pFechaInicioEjec": "01/01/1800",
        "pFechaFinEjec": "01/01/3000",
    }


class ContraloriaClient:
    def __init__(
        self,
        department: str = "CUSCO",
        delay_seconds: float = 0.4,
        timeout_seconds: float = 45,
        retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.department = department
        self.delay_seconds = max(0, delay_seconds)
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; CGRPublicDataResearch/1.0)",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://buscadorinformes.contraloria.gob.pe/BuscadorCGR/Informes/Avanzado.html",
            }
        )
        self._requested_page = False

    def fetch_page(self, page_number: int, page_size: int) -> list[dict[str, Any]]:
        if self._requested_page and self.delay_seconds:
            time.sleep(self.delay_seconds)
        self._requested_page = True

        params = build_params(self.department, page_number, page_size)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    SEARCH_URL, params=params, timeout=self.timeout_seconds
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise ScraperError("El endpoint no devolvió una lista de informes.")
                return [record for record in payload if isinstance(record, dict)]
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))

        raise ScraperError(f"No se pudo consultar la página {page_number}: {last_error}")
