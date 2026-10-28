import json
from pathlib import Path

from ocr_characterization import cli
from ocr_characterization.characterize import characterize_markdown
from ocr_characterization.easyocr_backend import extract_pdf_text as extract_easyocr_pdf_text
from ocr_characterization.paddle_backend import extract_pdf_text
from ocr_characterization.pytesseract_backend import (
    extract_pdf_text as extract_pytesseract_pdf_text,
    tesseract_language,
)
from ocr_characterization.schema import SITUATION_FIELDS


def test_extract_pdf_text_keeps_page_boundaries():
    class Result:
        def __init__(self, texts):
            self.json = {"res": {"rec_texts": texts}}

    class FakeOCR:
        def predict(self, image_path):
            return [Result([Path(image_path).stem, "texto reconocido"])]

    response = extract_pdf_text(FakeOCR(), [Path("page_0001.png"), Path("page_0002.png")])

    assert "[OCR_PAGE:1]" in response
    assert "[OCR_PAGE:2]" in response
    assert "page_0001\ntexto reconocido" in response


def test_easyocr_extract_pdf_text_keeps_page_boundaries():
    class FakeOCR:
        def readtext(self, image_path, detail, paragraph):
            assert detail == 0
            assert paragraph is False
            return [Path(image_path).stem, "texto reconocido"]

    response = extract_easyocr_pdf_text(
        FakeOCR(), [Path("page_0001.png"), Path("page_0002.png")]
    )

    assert "[OCR_PAGE:1]" in response
    assert "[OCR_PAGE:2]" in response
    assert "page_0001\ntexto reconocido" in response


def test_pytesseract_extract_pdf_text_keeps_page_boundaries():
    class FakeOCR:
        def image_to_string(self, image_path, lang):
            assert lang == "spa"
            return f" {Path(image_path).stem} texto reconocido \n"

    response = extract_pytesseract_pdf_text(
        FakeOCR(), [Path("page_0001.png"), Path("page_0002.png")]
    )

    assert "[OCR_PAGE:1]" in response
    assert "[OCR_PAGE:2]" in response
    assert "page_0001 texto reconocido" in response
    assert tesseract_language("es+en") == "spa+eng"


def test_characterize_ocr_text_extracts_page_citations():
    result = characterize_markdown(
        "[OCR_PAGE:1]\n\nInforme N.° 011-2021-CG\n2021\n"
        "[OCR_PAGE:2]\n\nSituación adversa N.° 1: Retraso de obra\n"
        "Condición: La obra presenta un retraso significativo en su ejecución.",
        "informe.pdf",
    )

    assert result["metadatos"]["id_informe"] == "011-2021-CG"
    assert result["citas_metadatos"]["id_informe"]["pagina"] == 1
    assert result["situaciones_adversas"][0]["num_situacion_adversa"] == 1
    assert result["situaciones_adversas"][0]["citas"]["titulo_situacion_adversa"]["pagina"] == 2
    assert len(SITUATION_FIELDS) == 13


def test_cli_runs_paddleocr_and_writes_characterization(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pdf_path = input_dir / "informe.pdf"
    pdf_path.touch()
    image_path = tmp_path / "page_0001.png"

    class FakeResult:
        json = {"res": {"rec_texts": ["Informe N.° 011-2021-CG"]}}

    class FakeOCR:
        def __init__(self, **kwargs):
            assert kwargs["lang"] == "es"
            assert kwargs["device"] == "cpu"

        def predict(self, _image):
            return [FakeResult()]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakeOCR))
    monkeypatch.setattr(cli, "render_pdf", lambda *_args, **_kwargs: [image_path])
    monkeypatch.setattr(cli.sys, "version_info", (3, 12, 0))

    status = cli.main([
        "--input-dir", str(input_dir),
        "--files", "informe.pdf",
        "--output-dir", str(tmp_path / "outputs"),
    ])

    assert status == 0
    assert (tmp_path / "outputs" / "informe.json").is_file()
    output = json.loads((tmp_path / "outputs" / "informe.json").read_text())
    assert output["metadatos"]["id_informe"] == "011-2021-CG"


def test_cli_runs_easyocr_and_writes_characterization(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "informe.pdf").touch()
    image_path = tmp_path / "page_0001.png"

    class FakeReader:
        def __init__(self, languages, gpu):
            assert languages == ["es"]
            assert gpu is True

        def readtext(self, _image, detail, paragraph):
            assert detail == 0
            assert paragraph is False
            return ["Informe N.° 011-2021-CG"]

    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=FakeReader))
    monkeypatch.setattr(cli, "render_pdf", lambda *_args, **_kwargs: [image_path])

    status = cli.main([
        "--input-dir", str(input_dir),
        "--files", "informe.pdf",
        "--output-dir", str(tmp_path / "outputs"),
        "--backend", "easyocr",
        "--device", "gpu",
    ])

    assert status == 0
    output = json.loads((tmp_path / "outputs" / "informe.json").read_text())
    assert output["metadatos"]["id_informe"] == "011-2021-CG"


def test_cli_runs_pytesseract_and_writes_characterization(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "informe.pdf").touch()
    image_path = tmp_path / "page_0001.png"

    def image_to_string(_image, lang):
        assert lang == "spa"
        return "Informe N.° 011-2021-CG"

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(image_to_string=image_to_string),
    )
    monkeypatch.setattr(cli, "render_pdf", lambda *_args, **_kwargs: [image_path])

    status = cli.main([
        "--input-dir", str(input_dir),
        "--files", "informe.pdf",
        "--output-dir", str(tmp_path / "outputs"),
        "--backend", "pytesseract",
    ])

    assert status == 0
    output = json.loads((tmp_path / "outputs" / "informe.json").read_text())
    assert output["metadatos"]["id_informe"] == "011-2021-CG"


def test_cli_rejects_gpu_for_pytesseract(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "informe.pdf").touch()

    status = cli.main([
        "--input-dir", str(input_dir),
        "--files", "informe.pdf",
        "--backend", "pytesseract",
        "--device", "gpu",
    ])

    assert status == 2
    assert "solo admite --device cpu" in capsys.readouterr().err


def test_cli_explains_missing_paddle_engine(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import sys

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "informe.pdf").touch()

    class MissingEngineOCR:
        def __init__(self, **_kwargs):
            raise RuntimeError("Engine 'paddle_static' is unavailable because dependency 'paddlepaddle' is not installed.")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=MissingEngineOCR))
    monkeypatch.setattr(cli.sys, "version_info", (3, 12, 0))

    status = cli.main([
        "--input-dir", str(input_dir),
        "--files", "informe.pdf",
    ])

    assert status == 1
    error = capsys.readouterr().err
    assert "Falta el motor paddlepaddle" in error
    assert "python -m pip install -e '.[cpu]' --extra-index-url" in error