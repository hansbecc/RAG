# Contraloría Cusco: extracción y RAG

Proyecto en dos etapas: recolecta los metadatos y PDFs públicos del buscador de la Contraloría y habilita consultas analíticas con LlamaIndex y Qdrant. La arquitectura y las decisiones de producción están en [docs/architecture.md](docs/architecture.md).

## Requisitos

- Python 3.11 o 3.12 recomendado para compatibilidad de PyMuPDF, FastEmbed y LlamaIndex.
- Docker Compose para el perfil local de Qdrant.
- Claves de OpenAI y Cohere para generación, embeddings y reranking.

## Scraper

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m scraper_cgr.cli --limit 3000 --download-pdfs
```

Guarda metadatos en `data/informes_cusco.csv` y `data/informes_cusco.jsonl`; los PDF se descargan en `data/pdfs/`.

## RAG local

Instala dependencias RAG, levanta Qdrant y exporta las claves:

```bash
python -m pip install -e ".[rag]"
docker compose up -d qdrant
export OPENAI_API_KEY="..."
export COHERE_API_KEY="..."
```

En Bash, usa `set -a; source .env; set +a` después de copiar `.env.example` a `.env` y completar claves. Nunca guardes secretos en control de versiones.

Indexa los PDFs y pregunta:

```bash
python -m rag_cgr.cli ingest --pdf-dir data/pdfs
python -m rag_cgr.cli ask "¿Qué patrones de retrasos y sobrecostos aparecen en las obras auditadas?"
```

La ingesta usa PyMuPDF por defecto. Para informes escaneados o tablas complejas, instala el extra y selecciona LlamaParse:

```bash
python -m pip install -e ".[rag,rag-ocr]"
export PDF_PARSER=llamaparse
export LLAMA_CLOUD_API_KEY="..."
python -m rag_cgr.cli ingest
```

Sin Cohere se puede desactivar el reranker con `RAG_RERANKER=none`; la búsqueda híbrida Qdrant permanece activa.

## Evaluación y pruebas

```bash
python -m unittest discover -s tests -v
```

El módulo `rag_cgr.evaluate` ofrece evaluadores LlamaIndex para faithfulness y relevancy. El documento de arquitectura describe el conjunto de preguntas etiquetadas, métricas de recuperación, umbrales y evaluación Ragas para un despliegue productivo.

## Configuración relevante

La configuración se lee desde variables de entorno; `.env.example` contiene los valores recomendados. El índice local de docstore se guarda en `data/rag/`; los embeddings y payloads viven en Qdrant. El scraper y el paquete RAG pueden utilizarse de forma independiente.

Respeta las condiciones y límites del buscador público. Este proyecto no intenta acceder a información reservada, secreta ni confidencial.