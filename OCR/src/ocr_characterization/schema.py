"""Esquema de salida alineado con la plantilla de caracterizacion."""

from __future__ import annotations

METADATA_FIELDS = (
    "id_informe",
    "anio_emision",
    "organo_control",
    "entidad_ejecutora",
    "provincia",
    "distrito",
    "cui_snip",
    "nombre_proyecto",
    "sector_tipologia_pip",
    "monto_inversion_soles",
    "modalidad_ejecucion",
    "modalidad_control",
    "fase_ejecucion_pip",
    "hito_control",
)

SITUATION_FIELDS = (
    "num_situacion_adversa",
    "titulo_situacion_adversa",
    "condicion_texto",
    "criterio_normativo_texto",
    "riesgo_efecto_texto",
    "proceso_componente_afectado",
    "evidencia_soporte",
    "clasificacion_anexo4_cgr",
    "subcategoria_anexo4",
    "monto_asociado_consecuencia",
    "cluster_tema_lda",
    "factor_asociado_inferido",
    "estado_situacion_adversa",
)

INFERRED_FIELDS = frozenset(
    {
        "clasificacion_anexo4_cgr",
        "subcategoria_anexo4",
        "cluster_tema_lda",
        "factor_asociado_inferido",
    }
)


def empty_result() -> dict:
    return {
        "metadatos": {field: None for field in METADATA_FIELDS},
        "situaciones_adversas": [],
        "observaciones": [],
    }