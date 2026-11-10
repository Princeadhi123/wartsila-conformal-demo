"""Smoke test: every app page renders without exceptions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _run_page(mod_name, root):
    # AppTest.from_function extracts this function's source into a temp
    # script, so it must be self-contained: all imports inside the body.
    import sys
    import importlib
    for p in (root + "/src", root + "/app"):
        if p not in sys.path:
            sys.path.insert(0, p)
    importlib.import_module(mod_name).show()


@pytest.mark.parametrize("page", [
    "page_overview", "page_marine", "page_energy", "page_trust", "page_pilot",
])
def test_page_renders(page):
    at = AppTest.from_function(
        _run_page, kwargs={"mod_name": page, "root": str(ROOT)})
    at.run(timeout=180)
    assert not at.exception, f"{page}: {at.exception}"


def test_full_app_runs():
    """The real entry point (st.navigation) renders its default page.

    at.switch_page only supports file-based pages/* multipage apps, not
    st.Page(function) navigation, so the other pages are covered by the
    per-page wrapper tests above.
    """
    at = AppTest.from_file(str(ROOT / "app" / "app.py"))
    at.run(timeout=180)
    assert not at.exception, at.exception
