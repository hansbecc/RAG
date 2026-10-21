"""Captura la página avanzada para documentar los puntos de extracción."""

from pathlib import Path

from playwright.sync_api import sync_playwright


PAGE_URL = "https://buscadorinformes.contraloria.gob.pe/BuscadorCGR/Informes/Avanzado.html"


def main() -> None:
    output = Path("screenshots/contraloria_avanzado.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=90000)
        page.get_by_role("button", name="REGIÓN").click()
        with page.expect_response(lambda response: "loadInformesElastic" in response.url):
            page.get_by_role("checkbox", name="CUSCO").check()
        page.locator("table tbody tr").first.wait_for()
        page.screenshot(path=str(output), full_page=True)
        browser.close()
    print(f"Captura guardada en {output}")


if __name__ == "__main__":
    main()
