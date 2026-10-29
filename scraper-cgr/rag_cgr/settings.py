"""Configuración tipada del servicio RAG, tomada del entorno del proceso."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RAGSettings:
    pdf_dir: Path = Path("data/pdfs")
    persist_dir: Path = Path("data/rag-ollama-nomic")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "contraloria_cusco_ollama_nomic768_bm25v2"
    llm_provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    llm_model: str = "qwen2.5:3b"
    openai_api_key: str | None = None
    reranker: str = "none"
    cohere_api_key: str | None = None
    rerank_model: str = "rerank-v3.5"
    dense_top_k: int = 40
    sparse_top_k: int = 40
    fusion_top_k: int = 24
    rerank_top_n: int = 8
    hybrid_alpha: float = 0.5
    parser: str = "pymupdf"
    llama_cloud_api_key: str | None = None
    ocr_lang: str = "es"
    ocr_device: str = "cpu"
    ocr_dpi: int = 200

    @classmethod
    def from_env(cls) -> "RAGSettings":
        provider = os.getenv("RAG_PROVIDER", "ollama").casefold()
        is_ollama = provider == "ollama"
        default_dimensions = "768" if is_ollama else "1536"
        default_embedding = "nomic-embed-text" if is_ollama else "text-embedding-3-small"
        default_llm = "qwen2.5:3b" if is_ollama else "gpt-4o"
        default_collection = (
            "contraloria_cusco_ollama_nomic768_bm25v2"
            if is_ollama
            else "contraloria_cusco_openai_1536"
        )
        default_persist_dir = "data/rag-ollama-nomic" if is_ollama else "data/rag-openai"
        return cls(
            pdf_dir=Path(os.getenv("RAG_PDF_DIR", "data/pdfs")),
            persist_dir=Path(os.getenv("RAG_PERSIST_DIR", default_persist_dir)),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv(
                "QDRANT_COLLECTION", default_collection
            ),
            llm_provider=provider,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            embedding_model=os.getenv("EMBEDDING_MODEL", default_embedding),
            embedding_dimensions=int(
                os.getenv("EMBEDDING_DIMENSIONS", default_dimensions)
            ),
            llm_model=os.getenv("LLM_MODEL", default_llm),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            reranker=os.getenv("RAG_RERANKER", "none").casefold(),
            cohere_api_key=os.getenv("COHERE_API_KEY") or None,
            rerank_model=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            dense_top_k=int(os.getenv("RAG_DENSE_TOP_K", "40")),
            sparse_top_k=int(os.getenv("RAG_SPARSE_TOP_K", "40")),
            fusion_top_k=int(os.getenv("RAG_FUSION_TOP_K", "24")),
            rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "8")),
            hybrid_alpha=float(os.getenv("RAG_HYBRID_ALPHA", "0.5")),
            parser=os.getenv("PDF_PARSER", "pymupdf").casefold(),
            llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY") or None,
            ocr_lang=os.getenv("PADDLEOCR_LANG", "es"),
            ocr_device=os.getenv("PADDLEOCR_DEVICE", "cpu").casefold(),
            ocr_dpi=int(os.getenv("PADDLEOCR_DPI", "200")),
        )

    def validate(
        self,
        *,
        require_llm: bool = True,
        require_reranker: bool = True,
    ) -> None:
        if self.llm_provider not in {"ollama", "openai"}:
            raise ValueError("RAG_PROVIDER debe ser 'ollama' o 'openai'.")
        if require_llm and self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("Define OPENAI_API_KEY para usar RAG_PROVIDER=openai.")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("RAG_HYBRID_ALPHA debe estar entre 0 y 1.")
        if min(self.dense_top_k, self.sparse_top_k, self.fusion_top_k, self.rerank_top_n) < 1:
            raise ValueError("Los top-k del pipeline deben ser mayores que cero.")
        if require_reranker and self.reranker == "cohere" and not self.cohere_api_key:
            raise ValueError("Define COHERE_API_KEY o configura RAG_RERANKER=none.")
        if self.parser == "llamaparse" and not self.llama_cloud_api_key:
            raise ValueError("Define LLAMA_CLOUD_API_KEY para usar LlamaParse.")
        if self.parser not in {"pymupdf", "llamaparse", "paddleocr"}:
            raise ValueError("PDF_PARSER debe ser 'pymupdf', 'llamaparse' o 'paddleocr'.")
        if self.ocr_device not in {"cpu", "gpu"}:
            raise ValueError("PADDLEOCR_DEVICE debe ser 'cpu' o 'gpu'.")
        if not 72 <= self.ocr_dpi <= 400:
            raise ValueError("PADDLEOCR_DPI debe estar entre 72 y 400.")
        if self.reranker not in {"cohere", "none"}:
            raise ValueError("RAG_RERANKER debe ser 'cohere' o 'none'.")
        if self.embedding_dimensions < 1:
            raise ValueError("EMBEDDING_DIMENSIONS debe ser mayor que cero.")
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_URL debe ser una URL HTTP/HTTPS.")
