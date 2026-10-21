"""Filtros de texto para identificar informes de obras y proyectos."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Any


DEFAULT_KEYWORDS = ("obra", "obras", "proyecto", "proyectos", "inversion")


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def matches_keywords(
    report: dict[str, Any], keywords: Iterable[str] = DEFAULT_KEYWORDS
) -> bool:
    title = normalize_text(str(report.get("Descripcion", "")))
    normalized_keywords = (normalize_text(word.strip()) for word in keywords)
    return any(keyword and keyword in title for keyword in normalized_keywords)
