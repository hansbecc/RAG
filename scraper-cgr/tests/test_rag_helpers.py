import unittest
from pathlib import Path

from rag_cgr.pipeline import (
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
    def test_local_settings_can_disable_external_reranker(self):
        settings = RAGSettings(reranker="none", openai_api_key="test")
        settings.validate()
        self.assertEqual(settings.persist_dir, Path("data/rag"))

    def test_ingestion_does_not_require_reranker_credentials(self):
        settings = RAGSettings(openai_api_key="test", reranker="cohere")
        settings.validate(require_reranker=False)


if __name__ == "__main__":
    unittest.main()
