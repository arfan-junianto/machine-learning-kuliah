from html.parser import HTMLParser
from pathlib import Path


PRESENTATION_DIR = Path(__file__).resolve().parents[1]
PRESENTATION = PRESENTATION_DIR / "Presentasi-Repository-Machine-Learning.html"


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
