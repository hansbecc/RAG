"""Configuración tipada del servicio RAG, tomada del entorno del proceso."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RAGSettings:
    pdf_dir: Path = Path("data/pdfs")
    persist_dir: Path = Path("data/rag")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "contraloria_cusco_reports"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o"
    openai_api_key: str | None = None
    reranker: str = "cohere"
    cohere_api_key: str | None = None
    rerank_model: str = "rerank-v3.5"
    dense_top_k: int = 40
    sparse_top_k: int = 40
    fusion_top_k: int = 24
    rerank_top_n: int = 8
    hybrid_alpha: float = 0.5
    parser: str = "pymupdf"
    llama_cloud_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "RAGSettings":
        return cls(
            pdf_dir=Path(os.getenv("RAG_PDF_DIR", "data/pdfs")),
            persist_dir=Path(os.getenv("RAG_PERSIST_DIR", "data/rag")),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv(
                "QDRANT_COLLECTION", "contraloria_cusco_reports"
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            reranker=os.getenv("RAG_RERANKER", "cohere").casefold(),
            cohere_api_key=os.getenv("COHERE_API_KEY") or None,
            rerank_model=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            dense_top_k=int(os.getenv("RAG_DENSE_TOP_K", "40")),
            sparse_top_k=int(os.getenv("RAG_SPARSE_TOP_K", "40")),
            fusion_top_k=int(os.getenv("RAG_FUSION_TOP_K", "24")),
            rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "8")),
            hybrid_alpha=float(os.getenv("RAG_HYBRID_ALPHA", "0.5")),
            parser=os.getenv("PDF_PARSER", "pymupdf").casefold(),
            llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY") or None,
        )

    def validate(
        self,
        *,
        require_llm: bool = True,
        require_reranker: bool = True,
    ) -> None:
        if require_llm and not self.openai_api_key:
            raise ValueError("Define OPENAI_API_KEY antes de usar el pipeline RAG.")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("RAG_HYBRID_ALPHA debe estar entre 0 y 1.")
        if min(self.dense_top_k, self.sparse_top_k, self.fusion_top_k, self.rerank_top_n) < 1:
            raise ValueError("Los top-k del pipeline deben ser mayores que cero.")
        if require_reranker and self.reranker == "cohere" and not self.cohere_api_key:
            raise ValueError("Define COHERE_API_KEY o configura RAG_RERANKER=none.")
        if self.parser == "llamaparse" and not self.llama_cloud_api_key:
            raise ValueError("Define LLAMA_CLOUD_API_KEY para usar LlamaParse.")
        if self.parser not in {"pymupdf", "llamaparse"}:
            raise ValueError("PDF_PARSER debe ser 'pymupdf' o 'llamaparse'.")
        if self.reranker not in {"cohere", "none"}:
            raise ValueError("RAG_RERANKER debe ser 'cohere' o 'none'.")
        if self.embedding_dimensions < 1:
            raise ValueError("EMBEDDING_DIMENSIONS debe ser mayor que cero.")
