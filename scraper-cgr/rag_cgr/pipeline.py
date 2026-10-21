"""Ingesta, búsqueda híbrida, fusión, reranking y generación de reportes."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_cgr.settings import RAGSettings

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.query_engine import RetrieverQueryEngine

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un analista senior de control gubernamental y redactas reportes
analíticos sustentados exclusivamente en el contexto recuperado.

Reglas obligatorias:
1. No inventes hechos, cifras, fechas, entidades, causalidad ni conclusiones. Si la
   evidencia no está en el contexto, decláralo expresamente.
2. Toda afirmación factual debe llevar una cita en línea exactamente con este
   formato: [Doc: nombre.pdf, Pág. X]. Copia el nombre y la página del marcador de
   procedencia del fragmento; no inventes citas ni páginas.
3. Separa los hechos documentados de las inferencias. Explica las limitaciones,
   contradicciones y datos faltantes relevantes.
4. Para cifras, indica unidad, periodo, denominador y operación cuando se puedan
   determinar desde las fuentes. No sumes importes de universos o periodos distintos.
5. Responde en español y Markdown con estas secciones: Resumen ejecutivo,
   Hallazgos clave, Métricas cuantitativas, Limitaciones y Bibliografía.
6. En Bibliografía, enumera solo los documentos que realmente sustentan la
   respuesta, en el formato [Doc: nombre.pdf, Pág. X].
"""


def citation_label(metadata: dict[str, Any]) -> str:
    """Devuelve una etiqueta de cita basada en el nombre y número de página."""
    filename = str(metadata.get("file_name") or metadata.get("filename") or "documento.pdf")
    page = metadata.get("page_number", metadata.get("page_label", "?"))
    page_text = str(page).strip()
    if page_text.isdigit():
        page_text = str(int(page_text))
    return f"[Doc: {Path(filename).name}, Pág. {page_text or '?'}]"


def _metadata_for_pdf(metadata: dict[str, Any], path: Path, fallback_page: int) -> dict[str, Any]:
    result = dict(metadata)
    raw_page = result.get("page_label", result.get("page_number", fallback_page))
    try:
        page_number = int(str(raw_page).strip())
    except (TypeError, ValueError):
        page_number = fallback_page
    result.update(
        {
            "file_name": path.name,
            "source_path": str(path.resolve()),
            "page_number": page_number,
            "page_label": str(page_number),
            "content_type": "pdf_page",
            "section_path": str(result.get("section_path", "")),
        }
    )
    return result


def _load_pdf(path: Path, settings: RAGSettings) -> list[Any]:
    if settings.parser == "llamaparse":
        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="markdown",
            parsing_instruction=(
                "Preserva títulos, tablas, notas al pie, códigos de informe y cifras. "
                "Mantén una página por unidad y no completes información ilegible."
            ),
        )
        documents = parser.load_data(str(path))
    elif settings.parser == "pymupdf":
        from llama_index.readers.file import PyMuPDFReader

        documents = PyMuPDFReader().load(file_path=path)
    else:
        raise ValueError("PDF_PARSER debe ser 'pymupdf' o 'llamaparse'.")

    normalized: list[Any] = []
    for index, document in enumerate(documents, start=1):
        document.metadata = _metadata_for_pdf(document.metadata, path, index)
        normalized.append(document)
    return normalized


def _create_qdrant(settings: RAGSettings) -> tuple[Any, Any, Any]:
    from qdrant_client import AsyncQdrantClient, QdrantClient, models
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    async_client = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key
    )
    if not client.collection_exists(settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "text-dense": models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
    for field_name, schema in (
        ("file_name", models.PayloadSchemaType.KEYWORD),
        ("page_number", models.PayloadSchemaType.INTEGER),
        ("section_path", models.PayloadSchemaType.KEYWORD),
        ("content_type", models.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(
                collection_name=settings.collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        except Exception as error:
            LOGGER.debug("No se pudo crear índice de payload %s: %s", field_name, error)

    vector_store = QdrantVectorStore(
        collection_name=settings.collection_name,
        client=client,
        aclient=async_client,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        dense_vector_name="text-dense",
        sparse_vector_name="text-sparse",
        batch_size=32,
    )
    return client, async_client, vector_store


def _configure_models(settings: RAGSettings) -> Any:
    from llama_index.core import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.openai_api_key,
    )
    llm = OpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
        system_prompt=SYSTEM_PROMPT,
    )
    Settings.llm = llm
    return llm


@dataclass(slots=True)
class RAGPipeline:
    settings: RAGSettings
    client: Any
    async_client: Any
    vector_store: Any
    index: "VectorStoreIndex | None" = None
    query_engine: "RetrieverQueryEngine | None" = None

    @classmethod
    def connect(
        cls,
        settings: RAGSettings | None = None,
        *,
        require_reranker: bool = True,
    ) -> "RAGPipeline":
        selected = settings or RAGSettings.from_env()
        selected.validate(require_reranker=require_reranker)
        _configure_models(selected)
        client, async_client, vector_store = _create_qdrant(selected)
        return cls(selected, client, async_client, vector_store)

    async def ingest_directory(self, pdf_dir: Path | None = None) -> int:
        """Indexa PDFs uno por uno, cediendo parsing e inserción bloqueantes a threads."""
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

        directory = pdf_dir or self.settings.pdf_dir
        pdf_paths = sorted(directory.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No se encontraron PDFs en {directory}.")

        self.settings.persist_dir.mkdir(parents=True, exist_ok=True)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        indexed_pages = 0

        for pdf_path in pdf_paths:
            documents = await asyncio.to_thread(_load_pdf, pdf_path, self.settings)
            nodes = await asyncio.to_thread(parser.get_nodes_from_documents, documents)
            leaves = get_leaf_nodes(nodes)
            if not leaves:
                LOGGER.warning("El parser no produjo nodos hoja para %s", pdf_path.name)
                continue

            await asyncio.to_thread(storage_context.docstore.add_documents, nodes)
            if self.index is None:
                self.index = await asyncio.to_thread(
                    VectorStoreIndex,
                    leaves,
                    storage_context=storage_context,
                    show_progress=True,
                )
            else:
                await asyncio.to_thread(self.index.insert_nodes, leaves)
            indexed_pages += len(documents)
            LOGGER.info("Indexado %s (%d páginas)", pdf_path.name, len(documents))

        if self.index is None:
            raise ValueError("No se generaron nodos indexables a partir de los PDFs.")
        await asyncio.to_thread(
            storage_context.persist,
            persist_dir=str(self.settings.persist_dir),
        )
        return indexed_pages

    def load_index(self) -> None:
        """Carga el docstore jerárquico local junto a los vectores persistidos en Qdrant."""
        from llama_index.core import StorageContext, load_index_from_storage

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=str(self.settings.persist_dir),
        )
        self.index = load_index_from_storage(storage_context)

    def build_query_engine(self) -> None:
        if self.index is None:
            self.load_index()

        from llama_index.core.postprocessor import NodePostprocessor
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
        from llama_index.core.response_synthesizers import ResponseMode
        from llama_index.core import get_response_synthesizer
        from llama_index.core.schema import NodeWithScore, QueryBundle

        class CitationMarkerPostprocessor(NodePostprocessor):
            def _postprocess_nodes(
                self,
                nodes: list[NodeWithScore],
                query_bundle: QueryBundle | None = None,
            ) -> list[NodeWithScore]:
                for item in nodes:
                    copied = item.model_copy(deep=True)
                    marker = citation_label(copied.node.metadata)
                    copied.node.set_content(f"{marker}\n{copied.node.get_content()}")
                    item.node = copied.node
                return nodes

        assert self.index is not None
        hybrid_retriever = self.index.as_retriever(
            vector_store_query_mode="hybrid",
            similarity_top_k=self.settings.fusion_top_k,
            sparse_top_k=self.settings.sparse_top_k,
            alpha=self.settings.hybrid_alpha,
        )
        fusion_retriever = QueryFusionRetriever(
            [hybrid_retriever],
            similarity_top_k=self.settings.fusion_top_k,
            num_queries=4,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=False,
        )
        storage_context = self.index.storage_context
        merging_retriever = AutoMergingRetriever(
            fusion_retriever,
            storage_context,
            verbose=False,
        )

        postprocessors: list[Any] = []
        if self.settings.reranker == "cohere":
            from llama_index.postprocessor.cohere_rerank import CohereRerank

            postprocessors.append(
                CohereRerank(
                    api_key=self.settings.cohere_api_key,
                    model=self.settings.rerank_model,
                    top_n=self.settings.rerank_top_n,
                )
            )
        elif self.settings.reranker != "none":
            raise ValueError("RAG_RERANKER debe ser 'cohere' o 'none'.")
        postprocessors.append(CitationMarkerPostprocessor())

        synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            llm=_configure_models(self.settings),
        )
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=merging_retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=postprocessors,
        )

    async def answer(self, question: str) -> str:
        if not question.strip():
            raise ValueError("La pregunta no puede estar vacía.")
        if self.query_engine is None:
            self.build_query_engine()
        assert self.query_engine is not None
        response = await self.query_engine.aquery(question)
        body = str(response).strip()
        citations = sorted(
            {
                citation_label(source.node.metadata)
                for source in response.source_nodes
            }
        )
        return ensure_bibliography(body, citations)

    async def close(self) -> None:
        await self.async_client.close()
        self.client.close()


async def ingest_pdf_directory(
    pdf_dir: Path | None = None,
    settings: RAGSettings | None = None,
) -> int:
    pipeline = RAGPipeline.connect(settings, require_reranker=False)
    try:
        return await pipeline.ingest_directory(pdf_dir)
    finally:
        await pipeline.close()


async def answer_question(
    question: str,
    settings: RAGSettings | None = None,
) -> str:
    pipeline = RAGPipeline.connect(settings)
    try:
        pipeline.load_index()
        pipeline.build_query_engine()
        return await pipeline.answer(question)
    finally:
        await pipeline.close()


def content_fingerprint(text: str, filename: str, page_number: int) -> str:
    """Clave estable útil para auditoría y deduplicación en procesos de ingestión."""
    payload = f"{filename}\0{page_number}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_report_markdown(markdown: str) -> str:
    """Normaliza saltos en el reporte sin alterar cifras, citas ni contenido factual."""
    return re.sub(r"\n{3,}", "\n\n", markdown)


def ensure_bibliography(markdown: str, citations: list[str]) -> str:
    """Añade citas recuperadas como bibliografía si el reporte aún no la incluye."""
    if not citations or "bibliografía" in markdown.casefold():
        return markdown
    entries = "\n".join(f"- {citation}" for citation in citations)
    return f"{markdown.rstrip()}\n\n## Bibliografía\n{entries}"
