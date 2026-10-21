import unittest

from scraper_cgr.client import build_params
from scraper_cgr.filters import matches_keywords, normalize_text


class FilterTests(unittest.TestCase):
    def test_matches_accents_and_case_in_report_title(self):
        report = {"Descripcion": "Mejoramiento del proyecto de inversión pública"}
        self.assertTrue(matches_keywords(report, ("inversion",)))

    def test_does_not_match_when_title_has_no_target_term(self):
        report = {"Descripcion": "Control de asistencia del personal"}
        self.assertFalse(matches_keywords(report))

    def test_normalizes_accents(self):
        self.assertEqual(normalize_text("Inversión Pública"), "inversion publica")


class RequestTests(unittest.TestCase):
    def test_builds_observed_cusco_pagination_parameters(self):
        params = build_params("Cusco", page_number=3, page_size=1000)
        self.assertEqual(params["Action"], "loadInformesElastic")
        self.assertEqual(params["PageNumber"], "3")
        self.assertEqual(params["PageSize"], "1000")
        self.assertEqual(params["pDepartamento"], '"CUSCO"')


if __name__ == "__main__":
    unittest.main()
