import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_cgr import cli
from rag_cgr.pipeline import (
    CHARACTERIZATION_PROMPT,
    _load_paddleocr_pdf,
    citation_label,
    content_fingerprint,
    ensure_bibliography,
    normalize_report_markdown,
)
from rag_cgr.settings import RAGSettings


class CitationTests(unittest.TestCase):
    def test_citation_uses_basename_and_page(self):
        self.assertEqual(
            citation_label({"file_name": "/tmp/auditoria.pdf", "page_number": 4}),
            "[Doc: auditoria.pdf, Pág. 4]",
        )

    def test_unknown_page_is_explicit(self):
        self.assertEqual(citation_label({"file_name": "x.pdf"}), "[Doc: x.pdf, Pág. ?]")


class UtilityTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_page_sensitive(self):
        first = content_fingerprint("texto", "archivo.pdf", 1)
        self.assertEqual(first, content_fingerprint("texto", "archivo.pdf", 1))
        self.assertNotEqual(first, content_fingerprint("texto", "archivo.pdf", 2))

    def test_report_whitespace_is_normalized_without_content_change(self):
        self.assertEqual(normalize_report_markdown("A\n\n\nB"), "A\n\nB")

    def test_bibliography_is_added_once(self):
        citation = "[Doc: informe.pdf, Pág. 2]"
        report = "## Hallazgos\nHallazgo documentado."
        with_bibliography = ensure_bibliography(report, [citation])
        self.assertIn(citation, with_bibliography)
        self.assertEqual(ensure_bibliography(with_bibliography, [citation]), with_bibliography)


class SettingsTests(unittest.TestCase):
    def test_default_settings_use_ollama_without_external_keys(self):
        settings = RAGSettings.from_env()

        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.embedding_model, "nomic-embed-text")
        self.assertEqual(settings.embedding_dimensions, 768)
        self.assertEqual(settings.reranker, "none")
        self.assertIsNone(settings.openai_api_key)
        self.assertIsNone(settings.cohere_api_key)
        settings.validate()

    def test_openai_profile_uses_separate_index_defaults(self):
        with patch.dict("os.environ", {"RAG_PROVIDER": "openai"}, clear=True):
            settings = RAGSettings.from_env()

        self.assertEqual(settings.embedding_dimensions, 1536)
        self.assertEqual(settings.collection_name, "contraloria_cusco_openai_1536")
        self.assertEqual(settings.persist_dir, Path("data/rag-openai"))

    def test_local_settings_can_disable_external_reranker(self):
        settings = RAGSettings(reranker="none", openai_api_key="test")
        settings.validate()
        self.assertEqual(settings.persist_dir, Path("data/rag-ollama-nomic"))

    def test_ingestion_does_not_require_reranker_credentials(self):
        settings = RAGSettings(openai_api_key="test", reranker="cohere")
        settings.validate(require_reranker=False)

    def test_paddleocr_configuration_is_validated(self):
        settings = RAGSettings(
            openai_api_key="test",
            reranker="none",
            parser="paddleocr",
            ocr_device="gpu",
            ocr_dpi=120,
        )
        settings.validate()
        with self.assertRaisesRegex(ValueError, "PADDLEOCR_DPI"):
            RAGSettings(
                openai_api_key="test", reranker="none", ocr_dpi=40
            ).validate()


class PaddleOCRIngestionTests(unittest.TestCase):
    def test_pdf_pages_become_cited_ocr_documents(self):
        class FakeDocument:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakePixmap:
            def save(self, path):
                Path(path).touch()

        class FakePage:
            def get_pixmap(self, matrix, alpha):
                self.matrix = matrix
                self.alpha = alpha
                return FakePixmap()

        class FakePdf:
            def __enter__(self):
                return [FakePage(), FakePage()]

            def __exit__(self, *_args):
                return False

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def predict(self, _image):
                class Result:
                    json = {"res": {"rec_texts": [" texto de prueba "]}}

                return [Result()]

        settings = RAGSettings(parser="paddleocr", ocr_dpi=100, ocr_device="cpu")
        fake_pymupdf = type(
            "FakePyMuPDF",
            (),
            {
                "open": staticmethod(lambda _path: FakePdf()),
                "Matrix": staticmethod(lambda *_scale: object()),
            },
        )
        fake_llama_index = type("FakeLlamaIndex", (), {"__path__": []})()
        fake_llama_core = type("FakeLlamaCore", (), {"Document": FakeDocument})()
        fake_paddleocr = type("FakePaddleOCRModule", (), {"PaddleOCR": FakePaddleOCR})()
        with patch.dict(
            "sys.modules",
            {
                "pymupdf": fake_pymupdf,
                "llama_index": fake_llama_index,
                "llama_index.core": fake_llama_core,
                "paddleocr": fake_paddleocr,
            },
        ):
            documents = _load_paddleocr_pdf(Path("informe.pdf"), settings)

        self.assertEqual(len(documents), 2)
        self.assertIn("[OCR_PAGE:1]", documents[0].text)
        self.assertIn("texto de prueba", documents[1].text)
        self.assertEqual(documents[1].metadata["page_number"], 2)
        self.assertEqual(documents[1].metadata["ocr_engine"], "paddleocr")


class CharacterizeCLITests(unittest.TestCase):
    def test_characterize_subcommand_takes_pdf_basename(self):
        args = cli.build_parser().parse_args(["characterize", "informe.pdf"])
        self.assertEqual(args.command, "characterize")
        self.assertEqual(args.file_name, "informe.pdf")
        self.assertIn("No completes", CHARACTERIZATION_PROMPT)


class LocalModelConfigurationTests(unittest.TestCase):
    def test_ollama_models_are_configured_without_api_keys(self):
        from rag_cgr.pipeline import _configure_models

        captured = {}

        class FakeEmbedding:
            def __init__(self, **kwargs):
                captured["embedding"] = kwargs

        class FakeLLM:
            def __init__(self, **kwargs):
                captured["llm"] = kwargs

        fake_settings = SimpleNamespace()
        fake_core = SimpleNamespace(Settings=fake_settings)
        with patch.dict(
            "sys.modules",
            {
                "llama_index.core": fake_core,
                "llama_index.embeddings": SimpleNamespace(__path__=[]),
                "llama_index.embeddings.ollama": SimpleNamespace(
                    OllamaEmbedding=FakeEmbedding
                ),
                "llama_index.llms": SimpleNamespace(__path__=[]),
                "llama_index.llms.ollama": SimpleNamespace(Ollama=FakeLLM),
            },
        ):
            llm = _configure_models(RAGSettings())

        self.assertIsInstance(llm, FakeLLM)
        self.assertEqual(captured["embedding"]["model_name"], "nomic-embed-text")
        self.assertEqual(captured["embedding"]["base_url"], "http://localhost:11434")
        self.assertEqual(captured["llm"]["model"], "qwen2.5:3b")
        self.assertEqual(captured["llm"]["temperature"], 0)
        self.assertNotIn("api_key", captured["llm"])


if __name__ == "__main__":
    unittest.main()
