from html.parser import HTMLParser
from pathlib import Path
import shutil

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


PRESENTATION_DIR = Path(__file__).resolve().parents[1]
PRESENTATION = PRESENTATION_DIR / "Presentasi-Repository-Machine-Learning.html"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ROOT_README = REPOSITORY_ROOT / "README.md"
PAGES_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "pages.yml"


class PresentationParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.html_lang = None
        self.slide_count = 0
        self.notes_count = 0
        self.external_resources: list[str] = []
        self.iframe_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        classes = set((attr.get("class") or "").split())

        if tag == "html":
            self.html_lang = attr.get("lang")
        if "slide" in classes:
            self.slide_count += 1
        if "speaker-notes" in classes:
            self.notes_count += 1
        if tag == "iframe":
            self.iframe_count += 1

        resource = None
        if tag in {"script", "img", "source", "video", "audio"}:
            resource = attr.get("src")
        elif tag == "link" and "stylesheet" in (attr.get("rel") or ""):
            resource = attr.get("href")
        if resource and not resource.startswith("data:"):
            self.external_resources.append(resource)


def read_presentation() -> str:
    assert PRESENTATION.exists(), "file HTML presentasi belum dibuat"
    return PRESENTATION.read_text(encoding="utf-8")


def parse_presentation(html: str) -> PresentationParser:
    parser = PresentationParser()
    parser.feed(html)
    return parser


def test_deck_has_ten_indonesian_slides_with_embedded_notes() -> None:
    html = read_presentation()
    parser = parse_presentation(html)

    assert parser.html_lang == "id"
    assert parser.slide_count == 10
    assert parser.notes_count == 10


def test_deck_contains_the_public_demo_routes_and_identity_placeholder() -> None:
    html = read_presentation()

    assert "https://github.com/arfan-junianto/machine-learning-kuliah" in html
    assert "https://colab.research.google.com/github/arfan-junianto/" in html
    assert "machine-learning-kuliah-edg2ydcbvp2yjsu7weh5dx.streamlit.app" in html
    assert "Arfan Junianto" in html
    assert "[NIM]" in html


def test_deck_keeps_medical_scope_and_course_citation_explicit() -> None:
    html = read_presentation().lower()

    assert "bukan diagnosis medis" in html
    assert "[ml-modul-01, hlm. 89–94]" in html


def test_deck_is_portable_without_embedded_pages_or_resource_dependencies() -> None:
    html = read_presentation()
    parser = parse_presentation(html)

    assert "/home/" not in html
    assert parser.iframe_count == 0
    assert parser.external_resources == []
    assert "fetch(" not in html


@pytest.fixture(scope="module")
def browser():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    chrome_binary = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome_binary:
        options.binary_location = chrome_binary

    driver = webdriver.Chrome(options=options)
    driver.get(PRESENTATION.as_uri())
    yield driver
    driver.quit()


def active_slide_index(browser) -> int:
    return browser.execute_script(
        "return [...document.querySelectorAll('.slide')].findIndex(s => s.classList.contains('active'));"
    )


def test_deck_starts_on_one_slide_and_supports_keyboard_navigation(browser) -> None:
    assert len(browser.find_elements(By.CSS_SELECTOR, ".slide.active")) == 1
    assert active_slide_index(browser) == 0

    browser.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_RIGHT)
    assert active_slide_index(browser) == 1

    browser.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    assert active_slide_index(browser) == 9

    browser.find_element(By.TAG_NAME, "body").send_keys(Keys.HOME)
    assert active_slide_index(browser) == 0


def test_mouse_controls_and_progress_report_the_current_slide(browser) -> None:
    browser.find_element(By.TAG_NAME, "body").send_keys(Keys.HOME)
    browser.find_element(By.ID, "next-slide").click()

    assert active_slide_index(browser) == 1
    assert browser.find_element(By.ID, "slide-status").text == "2 / 10"
    assert browser.find_element(By.ID, "slide-progress").get_attribute("style") == "width: 20%;"

    browser.find_element(By.ID, "previous-slide").click()
    assert active_slide_index(browser) == 0


def test_speaker_notes_are_hidden_by_default_and_toggle_with_n(browser) -> None:
    browser.refresh()
    notes = browser.find_element(By.ID, "notes-panel")

    assert notes.get_attribute("aria-hidden") == "true"
    assert not notes.is_displayed()

    browser.find_element(By.TAG_NAME, "body").send_keys("n")
    assert notes.get_attribute("aria-hidden") == "false"
    assert notes.is_displayed()
    assert "Selamat pagi/siang" in notes.text

    browser.find_element(By.TAG_NAME, "body").send_keys("n")
    assert not notes.is_displayed()


@pytest.mark.parametrize("width,height", [(1366, 768), (1920, 1080)])
def test_every_slide_fits_common_projector_viewports(browser, width: int, height: int) -> None:
    browser.set_window_size(width, height)
    browser.find_element(By.TAG_NAME, "body").send_keys(Keys.HOME)

    for index in range(10):
        if index:
            browser.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_RIGHT)
        dimensions = browser.execute_script(
            """
            const slide = document.querySelector('.slide.active');
            return {
              width: slide.clientWidth,
              height: slide.clientHeight,
              scrollWidth: slide.scrollWidth,
              scrollHeight: slide.scrollHeight
            };
            """
        )
        assert dimensions["scrollWidth"] <= dimensions["width"] + 1, f"slide {index + 1} overflow horizontal"
        assert dimensions["scrollHeight"] <= dimensions["height"] + 1, f"slide {index + 1} overflow vertical"


def test_root_readme_links_the_offline_repository_presentation() -> None:
    readme = ROOT_README.read_text(encoding="utf-8")

    assert "Presentasi Repository (offline)" in readme
    assert (
        "https://arfan-junianto.github.io/machine-learning-kuliah/"
        "Presentasi-Repository-Machine-Learning.html"
    ) in readme


def test_pages_workflow_publishes_assignment_and_visual_presentation() -> None:
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")

    assert '"coursework/presentations/repository-machine-learning/**"' in workflow
    assert "mkdir -p _site" in workflow
    assert "coursework/assignments/pertemuan-06/. _site/" in workflow
    assert (
        "coursework/presentations/repository-machine-learning/"
        "Presentasi-Repository-Machine-Learning.html" in workflow
    )
    assert "path: _site" in workflow
