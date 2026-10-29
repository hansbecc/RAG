# Contraloría Cusco: extracción y RAG

Proyecto en dos etapas: recolecta los metadatos y PDFs públicos del buscador de la Contraloría y habilita consultas analíticas con LlamaIndex y Qdrant. La arquitectura y las decisiones de producción están en [docs/architecture.md](docs/architecture.md).

## Requisitos

- Python 3.11 o 3.12 recomendado para compatibilidad de PyMuPDF, FastEmbed y LlamaIndex.
- Docker Compose para el perfil local de Qdrant.
- Ollama local para generación y embeddings; Docker Compose para Qdrant.

## Scraper

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m scraper_cgr.cli --limit 3000 --download-pdfs
```

Guarda metadatos en `data/informes_cusco.csv` y `data/informes_cusco.jsonl`; los PDF se descargan en `data/pdfs/`.

## RAG local

Instala dependencias RAG, instala Ollama y levanta Qdrant:

```bash
python -m pip install -e ".[rag]"
ollama serve  # terminal 1
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
docker compose up -d qdrant
```

Instala el servicio Ollama según tu sistema antes de iniciar esos comandos. Mantén `ollama serve` activo en otra terminal mientras descargas modelos y usas el RAG. Ollama sirve localmente en `http://localhost:11434`; no se requieren `OPENAI_API_KEY` ni `COHERE_API_KEY`. El reranker queda desactivado por defecto (`RAG_RERANKER=none`). Para el modo opcional OpenAI/Cohere, instala además `.[rag-openai,rag-cohere]`, configura `RAG_PROVIDER=openai`, las claves correspondientes, `RAG_RERANKER=cohere` y los valores de modelos/dimensiones OpenAI.

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

El reranker es opcional; con `RAG_RERANKER=none` la búsqueda híbrida de Qdrant permanece activa.

## Caracterización RAG con PaddleOCR

Para informes escaneados, configura la ingesta para que PaddleOCR reconozca cada página localmente antes de indexarla. Usa un entorno Python 3.10–3.12 y el wheel PaddlePaddle CPU o GPU compatible con el driver:

```bash
cd scraper-cgr
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -e '.[rag,rag-paddle-ocr]'
export PDF_PARSER=paddleocr
export PADDLEOCR_LANG=es
export PADDLEOCR_DEVICE=cpu
export PADDLEOCR_DPI=100
python -m rag_cgr.cli ingest --pdf-dir data/pdfs
python -m rag_cgr.cli characterize INFORME03.pdf
```

Para GPU, instala en lugar del wheel CPU un `paddlepaddle-gpu` compatible con el driver y CUDA siguiendo la [guía oficial](https://www.paddleocr.ai/latest/en/version3.x/paddlepaddle_installation.html), y cambia `PADDLEOCR_DEVICE` a `gpu`. El nombre para `characterize` debe coincidir con el nombre base del PDF ya indexado. El comando restringe la recuperación a ese informe y produce metadatos y situaciones adversas con evidencia/citas de página. Los valores ausentes se declaran como `No identificado`; clasificación Anexo 4, subcategorías y factores inferidos no se completan automáticamente. Revisa toda salida contra el informe original. La ingesta OCR respeta `PADDLEOCR_LANG`, `PADDLEOCR_DEVICE` y `PADDLEOCR_DPI`; si ya existe un índice PyMuPDF, reingresa los documentos a una colección nueva o limpia/versiona el índice antes de mezclar parsers.

El comando `characterize` usa Ollama local para embeddings y síntesis, Qdrant para recuperación y no requiere proveedor de reranking. Levanta Qdrant con `docker compose up -d qdrant` desde `scraper-cgr/`; Ollama y ambos modelos deben estar disponibles localmente. El perfil por defecto usa 768 dimensiones y una colección/docstore separados de cualquier índice OpenAI de 1536 dimensiones. La búsqueda sparse usa FastEmbed BM25 multilingüe con modificador IDF en Qdrant.

## Evaluación y pruebas

```bash
python -m unittest discover -s tests -v
```

El módulo `rag_cgr.evaluate` ofrece evaluadores LlamaIndex para faithfulness y relevancy. El documento de arquitectura describe el conjunto de preguntas etiquetadas, métricas de recuperación, umbrales y evaluación Ragas para un despliegue productivo.

## Configuración relevante

La configuración se lee desde variables de entorno; `scraper-cgr/.env.example` contiene los valores recomendados. El perfil Ollama guarda el docstore en `scraper-cgr/data/rag-ollama-nomic/` y los embeddings/payloads en su colección Qdrant dedicada. Scraper y RAG se pueden utilizar de forma independiente.

Respeta las condiciones y límites del buscador público. Este proyecto no intenta acceder a información reservada, secreta ni confidencial.