from pathlib import Path
import shutil

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


HTML_PATH = (
    Path(__file__).resolve().parents[1] / "Jawaban-Latihan-Kisi-kisi-UTS.html"
)


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
    yield driver
    driver.quit()


def load_layout(browser, width, media="screen"):
    browser.execute_cdp_cmd("Emulation.setEmulatedMedia", {"media": media})
    browser.set_window_size(width, 900)
    browser.get(HTML_PATH.as_uri())
    return browser.execute_script(
        """
        const section = document.querySelector('.section.first');
        const style = getComputedStyle(section);
        const rect = section.getBoundingClientRect();
        return {
          innerWidth: window.innerWidth,
          clientWidth: document.documentElement.clientWidth,
          documentWidth: document.documentElement.scrollWidth,
          left: rect.left,
          right: rect.right,
          width: rect.width,
          paddingLeft: parseFloat(style.paddingLeft),
          paddingRight: parseFloat(style.paddingRight),
          maxWidth: style.maxWidth,
          bodyBackground: getComputedStyle(document.body).backgroundColor,
        };
        """
    )


def test_desktop_content_is_centered_with_readable_gutters(browser):
    layout = load_layout(browser, 1440)

    assert layout["width"] <= 1180.5
    assert layout["left"] >= 100
    right_gutter = layout["clientWidth"] - layout["right"]
    assert abs(layout["left"] - right_gutter) <= 1
    assert layout["paddingLeft"] >= 18
    assert layout["paddingRight"] >= 18
    assert layout["bodyBackground"] != "rgb(255, 255, 255)"


def test_mobile_content_keeps_padding_without_horizontal_overflow(browser):
    layout = load_layout(browser, 390)

    assert layout["paddingLeft"] >= 18
    assert layout["paddingRight"] >= 18
    assert layout["documentWidth"] <= layout["clientWidth"]


def test_mobile_cover_keeps_readable_text_width(browser):
    browser.execute_cdp_cmd("Emulation.setEmulatedMedia", {"media": "screen"})
    browser.set_window_size(390, 900)
    browser.get(HTML_PATH.as_uri())
    layout = browser.execute_script(
        """
        const cover = document.querySelector('.cover');
        const subtitle = document.querySelector('.subtitle');
        const heading = document.querySelector('.cover h1');
        return {
          coverPaddingLeft: parseFloat(getComputedStyle(cover).paddingLeft),
          subtitleAlign: getComputedStyle(subtitle).textAlign,
          headingSize: parseFloat(getComputedStyle(heading).fontSize),
          documentWidth: document.documentElement.scrollWidth,
          clientWidth: document.documentElement.clientWidth,
        };
        """
    )

    assert layout["coverPaddingLeft"] <= 24
    assert layout["subtitleAlign"] == "left"
    assert layout["headingSize"] <= 40
    assert layout["documentWidth"] <= layout["clientWidth"]


def test_print_media_does_not_use_screen_container_spacing(browser):
    layout = load_layout(browser, 1440, media="print")

    assert layout["maxWidth"] == "none"
    assert layout["paddingLeft"] == 0
    assert layout["paddingRight"] == 0
    assert layout["bodyBackground"] == "rgb(255, 255, 255)"
