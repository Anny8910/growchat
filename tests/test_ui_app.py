"""Smoke tests that the Streamlit app actually renders.

`tests/test_ui.py` covers the view model; this file covers the drawing, because
a UI that raises on first run is the one failure the unit tests cannot see.
`AppTest` runs the real script headlessly, so no browser or server is needed.

No question is asked here — every one of these runs the app to its resting
state, so the suite still makes no LLM call.
"""

from __future__ import annotations

import pathlib

import pytest

streamlit = pytest.importorskip("streamlit")

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

# AppTest resolves a relative path against the calling file, not the cwd.
APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")


def _run(**kwargs):
    at = AppTest.from_file(APP, default_timeout=180, **kwargs)
    at.run()
    return at


def test_the_app_renders_without_raising():
    at = _run()
    assert at.exception == [], [str(e) for e in at.exception]


def test_the_disclaimer_is_rendered_from_config():
    import config

    at = _run()
    rendered = " ".join(c.value for c in at.caption)
    assert config.DISCLAIMER in rendered


def test_the_three_example_questions_are_offered():
    import config

    at = _run()
    labels = [b.label for b in at.button]
    for example in config.EXAMPLE_QUESTIONS:
        assert example in labels


def test_a_clear_chat_button_is_present():
    at = _run()
    assert "Clear chat" in [b.label for b in at.button]


def test_the_chat_input_is_available():
    at = _run()
    assert len(at.chat_input) == 1


def test_the_welcome_line_shows_on_an_empty_session():
    at = _run()
    assert at.chat_message == [], "a fresh session must not render a stale thread"
    body = " ".join(m.value for m in at.markdown)
    assert "Ask me about these five HDFC mutual fund schemes" in body


def test_a_missing_store_is_built_on_first_use(monkeypatch):
    """Streamlit Cloud has no build step, so the app builds the store itself.

    Without this the deploy shows every visitor "No vector store. Run python
    ingest.py first." on a machine they don't control.

    `ingest.ensure_store` is patched rather than `app.ensure_store` because
    AppTest re-executes app.py as __main__, so its `from ingest import` binding
    happens inside the run and only sees a patch applied to `ingest`.
    """
    import config
    import ingest

    state = {"built": False, "called": 0}

    def fake_ensure(*_a, **_k):
        state["called"] += 1
        state["built"] = True          # a successful build makes it appear
        return True, "store built"

    monkeypatch.setattr(config, "chroma_store_exists", lambda: state["built"])
    monkeypatch.setattr(ingest, "ensure_store", fake_ensure)

    at = _run()
    assert at.exception == []
    assert state["called"] == 1, "ensure_store was never called"
    assert not [e for e in at.error if "vector store" in e.value.lower()]
    assert at.chat_input, "the app should continue to a usable chat"


def test_a_failed_store_build_is_reported_instead_of_crashing(monkeypatch):
    """A setup mistake must be a sentence on screen, not a stack trace."""
    import config
    import ingest

    monkeypatch.setattr(config, "chroma_store_exists", lambda: False)
    monkeypatch.setattr(
        ingest, "ensure_store",
        lambda *_a, **_k: (False, "corpus is missing or malformed"),
    )

    at = _run()
    assert at.exception == []
    errors = " ".join(e.value for e in at.error)
    assert "corpus is missing or malformed" in errors
