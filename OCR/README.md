# Caracterizacion de informes CGR con PaddleOCR

Lee `data/INFORME03.pdf`, `data/INFORME04.pdf` y `data/INFORME05.pdf`, y genera una caracterizacion de metadatos y situaciones adversas con la estructura de `data/PLANTILLA.pdf`.

El proyecto renderiza cada pagina del PDF y reconoce el texto con PaddleOCR 3.x, EasyOCR o PyTesseract. A partir del texto reconocido, extrae campos documentales mediante reglas conservadoras y conserva numero de pagina y fragmentos de evidencia. Los motores OCR no son modelos generativos ni clasifican situaciones adversas; los campos que requieren interpretacion o inferencia se dejan como `null` para revision humana.

## Requisitos

- Python 3.10–3.12. PaddlePaddle no publica un wheel compatible con el Python 3.14 de este equipo.
- PaddleOCR 3.x y PaddlePaddle 3.x. La primera ejecucion descarga los modelos OCR seleccionados.
- EasyOCR es un backend opcional; requiere PyTorch y descarga sus modelos en la primera ejecucion.
- PyTesseract es un backend opcional y requiere el ejecutable Tesseract y los datos de idioma instalados en el sistema.
- CPU compatible para ejecutar localmente. GPU NVIDIA es opcional; requiere instalar un wheel `paddlepaddle-gpu` compatible con el driver/CUDA del equipo.
- Espacio suficiente para los modelos OCR y las imagenes temporales de las paginas.

Instala el proyecto desde `OCR/` con un entorno Python 3.12 administrado por `uv`. Si no tienes `uv` en Manjaro, instálalo primero con `sudo pacman -S uv`:

```bash
cd OCR
uv python install 3.12
uv venv --python 3.12 .venv-paddle
uv pip install --python .venv-paddle/bin/python -e '.[cpu,test]' --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ --extra-index-url https://pypi.org/simple
```

`uv pip` instala dentro del entorno indicado sin necesitar que exista `pip` en el venv. Comprueba la versión y ejecuta el proyecto con el intérprete de `.venv-paddle`; no reutilices `.venv` si fue creado con Python 3.14:

```bash
.venv-paddle/bin/python --version
.venv-paddle/bin/python -m ocr_characterization --dry-run
.venv-paddle/bin/python -m ocr_characterization
```

El modo CPU no necesita CUDA y es el valor predeterminado. Para habilitar GPU, instala manualmente el wheel `paddlepaddle-gpu` correspondiente a tu version de Python y driver siguiendo la [guia oficial de instalacion](https://www.paddleocr.ai/latest/en/version3.x/paddlepaddle_installation.html), y luego instala el proyecto sin el extra `cpu`. La GTX 1650 puede intentarse con `--device gpu`, pero con 4 GB la memoria puede ser limitada; si falla o se queda sin memoria, usa CPU y/o baja el DPI.

## Uso

Activa el entorno y valida los PDFs sin cargar PaddleOCR ni descargar modelos:

```bash
cd OCR
.venv-paddle/bin/python -m ocr_characterization --dry-run
```

Ejecuta la caracterizacion completa:

```bash
.venv-paddle/bin/python -m ocr_characterization
```

Por defecto, lee los tres informes de `data/` y escribe en `outputs/` un conjunto de archivos por informe:

- `INFORME03.response.txt`: texto OCR concatenado con marcadores de pagina.
- `INFORME03.json`: campos detectados, evidencia, paginas y marcas de revision humana.
- `INFORME03.csv`: una fila por campo caracterizado, con valor, evidencia y tipo de campo.
- `INFORME03.md`: tablas legibles de metadatos y situaciones adversas.

Repite para `INFORME04` e `INFORME05`. Puedes seleccionar archivos, idioma OCR, dispositivo y DPI:

```bash
.venv-paddle/bin/python -m ocr_characterization --input-dir data --output-dir outputs --files INFORME03.pdf INFORME04.pdf --lang es --device cpu --dpi 200
```

Para intentar usar GPU con PaddlePaddle GPU instalado:

```bash
.venv-paddle/bin/python -m ocr_characterization --device gpu
```

## EasyOCR

Instala EasyOCR en un entorno separado; no requiere PaddleOCR ni PaddlePaddle. EasyOCR instala PyTorch; para GPU, asegúrate de que la instalacion de PyTorch sea compatible con el driver NVIDIA del equipo.

```bash
cd OCR
uv venv --python 3.12 .venv-easyocr
uv pip install --python .venv-easyocr/bin/python -e '.[easyocr,test]'
.venv-easyocr/bin/python -m ocr_characterization --backend easyocr --device cpu --dpi 100 --output-dir outputs-easyocr
```

Para PaddleOCR en CPU, instala el extra `.[cpu,test]` desde el indice CPU de Paddle y PyPI; para PaddleOCR GPU, instala `paddlepaddle-gpu` compatible con el driver y luego `.[paddle,test]`.

Usa `--device gpu` si PyTorch detecta una GPU CUDA disponible. El backend acepta los mismos argumentos `--files`, `--lang` y `--dpi`, y produce los mismos archivos JSON, CSV, Markdown y texto OCR.

## PyTesseract

Instala el adaptador Python y el motor Tesseract junto con el paquete de idioma. En Manjaro/Arch, por ejemplo:

```bash
sudo pacman -S tesseract tesseract-data-spa
cd OCR
uv venv --python 3.12 .venv-tesseract
uv pip install --python .venv-tesseract/bin/python -e '.[pytesseract]'
.venv-tesseract/bin/python -m ocr_characterization --backend pytesseract --lang es --device cpu --dpi 200 --output-dir outputs-tesseract
```

El backend traduce `es`, `en`, `fr`, `de`, `it` y `pt` a los códigos de modelo de Tesseract (`spa`, `eng`, `fra`, `deu`, `ita`, `por`). También puedes indicar códigos instalados directamente o combinarlos, por ejemplo `--lang spa+eng`. PyTesseract utiliza CPU en esta implementación y no acepta `--device gpu`.

## Evaluacion y limites

La plantilla contiene caracterizaciones de referencia, pero no se toma como respuesta automatica ni como verdad de terreno. Revisa cada campo y cita contra el PDF original. PaddleOCR reconoce texto, pero no interpreta de forma fiable tablas complejas, relaciones entre secciones o categorias de control; los campos que requieren inferencia se dejan vacios. El proyecto no calcula una puntuacion de precision sin un conjunto de etiquetas humanas independientes.

Los modelos de PaddleOCR o EasyOCR se descargan en la primera ejecucion. PyTesseract utiliza los datos de idioma instalados en el sistema. La inferencia se ejecuta localmente; no se llama a un servicio OCR externo.

## Pruebas

```bash
uv pip install --python .venv-paddle/bin/python -e '.[cpu,test]' --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ --extra-index-url https://pypi.org/simple
.venv-paddle/bin/python -m pytest
```