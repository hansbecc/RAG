# Arquitectura RAG para informes de control

## 1. Ingesta, parsing y chunking

### Flujo

1. El scraper guarda los PDF y el JSONL/CSV con número de informe, entidad y URL. Mantener el identificador de informe en el catálogo permite auditar y volver a descargar una fuente.
2. PyMuPDF (`PyMuPDFReader`) extrae texto página a página. La ingesta normaliza `file_name`, `source_path`, `page_number`, `page_label`, `content_type` y `section_path` antes de generar nodos.
3. `HierarchicalNodeParser([2048, 512, 128])` produce niveles padre-hijo. Se indexan únicamente hojas; el resto se conserva en `SimpleDocumentStore`. `AutoMergingRetriever` amplía a nodos padre cuando suficientes hijos de una sección son relevantes.
4. Las operaciones de parseo e inserción síncronas se ejecutan con `asyncio.to_thread`; las respuestas usan `aquery` y los clientes Qdrant asíncrono y síncrono se cierran explícitamente.

### Calidad del parsing

PyMuPDF es la ruta económica para PDF con texto digital. Un informe de control con tablas de varias columnas, sellos, anexos escaneados o diagramas debe pasar por una cola de excepción: comprobar páginas con muy poco texto, reintentar con `PDF_PARSER=llamaparse` y registrar versión del parser, resultado y coste. LlamaParse entrega Markdown y suele preservar tablas mejor; sigue siendo necesaria una validación contra el PDF original. Para OCR local puede usarse PyMuPDF/Tesseract, siempre con español (`spa`) instalado. Ningún OCR debe transformar un dígito dudoso sin marcarlo como incierto.

Al mejorar el parser, normalizar encabezados y pies repetidos, mantener tablas como Markdown o filas estructuradas y registrar número de página impreso y número físico de página. Diagramas complejos requieren extracción visual multimodal y revisión humana; no se consideran cubiertos por el MVP textual.

### Chunking y contexto

La jerarquía mantiene relaciones padre-hijo (`NodeRelationship.PARENT`/`CHILD`) y metadatos de página. Los tamaños son puntos de partida, no constantes universales: medir recuperación de tablas, encabezados y conclusiones con un conjunto etiquetado antes de modificar el tamaño/overlap. Si se necesita seccionado semántico, probar `SemanticSplitterNodeParser` como experimento y comparar su latencia y recall con la jerarquía fija. Los límites de página son deliberados para que la cita sea precisa; no unir páginas cuando se pierda la atribución.

## 2. Qdrant y payload

`QdrantVectorStore(enable_hybrid=True, fastembed_sparse_model="Qdrant/bm25")` crea vectores densos y dispersos. El MVP usa embeddings `text-embedding-3-small` de 1.536 dimensiones y distancia Cosine. Cosine es una opción estable para embeddings normalizados; usar Dot solo si el modelo y la normalización se miden y la colección declara la misma distancia desde su creación. No cambiar dimensión, modelo ni configuración de vectores en una colección existente: crear una colección versionada y reindexar.

Payload recomendado: `file_name`, hash del archivo, número de informe, región, entidad, página, ruta de sección, tipo de contenido, versión del parser y fecha de ingesta. El MVP indexa los campos de filtro `file_name`, `page_number`, `section_path` y `content_type`; los metadatos se añaden a cada `TextNode` y Qdrant los conserva como payload. En producción, añadir índices de payload para filtros que se utilicen realmente y filtrar por región/año/entidad antes de recuperar.

La búsqueda híbrida de Qdrant fusiona dense + sparse mediante relative score fusion (`alpha=0.5` por defecto). `similarity_top_k` determina candidatos tras fusión y `sparse_top_k` define candidatos por canal, no deben confundirse. Fijar el modelo sparse desde el primer punto de indexado. Las consultas lexicográficas (código de informe, entidad o monto exacto) deben tener peso suficiente de sparse.

## 3. Recuperación avanzada

El flujo implementado es:

```text
Pregunta
  -> VectorIndexRetriever(mode="hybrid", dense + Qdrant BM25 sparse)
  -> QueryFusionRetriever(4 consultas, reciprocal-rank fusion)
  -> AutoMergingRetriever(hojas -> sección padre si aporta cobertura)
  -> CohereRerank(top_n=8)
  -> marcador de procedencia [Doc: ..., Pág. ...]
  -> ResponseSynthesizer(compact)
```

La expansión multi-query recupera formulaciones alternativas sin sustituir la pregunta original. RRF reduce la sensibilidad a las escalas de score entre queries. Cohere es un reranker gestionado; desactivarlo (`RAG_RERANKER=none`) reduce coste/latencia a cambio de precisión de orden. Si se adopta un cross-encoder local, comprobar memoria, idioma español y rendimiento del modelo con la evaluación propia. Un reranker no recupera evidencia que nunca llegó al conjunto de candidatos.

## 4. Síntesis, citas y prompt

El sistema usa `ResponseMode.COMPACT` para preguntas analíticas acotadas. Para reportes que exijan sintetizar un corpus amplio, usar `tree_summarize` sobre una selección documentada de fuentes, no sobre toda la colección sin trazabilidad. El LLM/embedding de este MVP es OpenAI; el proveedor se puede cambiar por un adaptador LlamaIndex (por ejemplo, Anthropic) manteniendo pruebas de citas y evaluación.

El marcador de cada nodo lleva nombre base y página, y el system prompt exige citas en el formato exacto `[Doc: nombre.pdf, Pág. X]` después de cada afirmación factual. Se agrega una bibliografía determinista con fuentes recuperadas; la bibliografía por sí sola no sustituye las citas en línea. Para producción, validar salida con regex contra el conjunto de páginas efectivamente recuperadas; ante citas desconocidas o afirmaciones sin cita, rechazar/regenerar o marcar el reporte para revisión humana.

El prompt exacto está en `rag_cgr.pipeline.SYSTEM_PROMPT`. Reglas: responder solo con evidencia, declarar vacíos y contradicciones, separar inferencias y hechos, no inventar cifras y producir Resumen ejecutivo, Hallazgos clave, Métricas cuantitativas, Limitaciones y Bibliografía.

## 5. Evaluación

`rag_cgr.evaluate.evaluate_response` utiliza `FaithfulnessEvaluator` y `RelevancyEvaluator` de LlamaIndex. Ejecutar ambos sobre un set de preguntas revisadas por analistas y comparar por parser/modelo/configuración. Registrar pregunta, respuesta, ids de página recuperados, versión de colección, prompt, latencia y coste. Además medir retrieval: `Recall@K`, `MRR`/`nDCG` frente a páginas oro, cobertura de cifras, exactitud de cita (documento y página) y tasa de abstención correcta. No usar el LLM evaluador como única señal de calidad.

Ragas es una alternativa de evaluación a nivel conjunto con `faithfulness`, `answer_relevancy`, `context_precision` y `context_recall`. Para usarlo, construir `SingleTurnSample` desde preguntas/respuestas etiquetadas y contexts tomados de `response.source_nodes`, declarar explícitamente los adaptadores LLM/embeddings y ejecutar fuera del camino de consulta. Mantener el split de evaluación fuera de las instrucciones del prompt y versionar el dataset para evitar regresiones silenciosas.

## 6. Operación y endurecimiento

- Qdrant expone puertos solo a localhost en `docker-compose.yml`; para despliegue remoto, TLS, API key, red privada, secretos en gestor de secretos y snapshots/backups probados.
- `data/rag/` conserva el docstore padre-hijo y Qdrant conserva vectores/payload. Ambos deben respaldarse y restaurarse en conjunto; sus versiones forman una misma versión de índice.
- El MVP es para una primera ingesta controlada. Antes de reingestas incrementales, implementar hash de PDF, upsert idempotente por página/nodo, borrado de versiones obsoletas y checkpoints transaccionales. No ejecutar varias ingestas concurrentes sobre la misma colección.
- Añadir límites de páginas/tamaño de PDF, antivirus si la fuente no es confiable, control de MIME, retries con backoff, métricas, logs JSON, trazas y rate limits del LLM/reranker. Los PDFs son entrada no confiable: tratar texto como datos, nunca como instrucciones del agente.
- Minimizar metadatos personales; las citas remiten a documentación pública y no reemplazan la validación legal o el juicio de auditoría.