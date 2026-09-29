"""Config and environment-loading tests.

Phase 1 setup, pinned. These are cheap to break: a `.gitignore` line edited to
make `git add -A` convenient, or a refactor that drops `load_dotenv`, both fail
silently and only surface as a missing key at the first real request.
"""

from __future__ import annotations

import os
import subprocess

import pytest
from dotenv import load_dotenv

import config

ROOT = config.ROOT


# --- .env.example -----------------------------------------------------------

def test_env_example_exists():
    assert (ROOT / ".env.example").is_file()


def test_env_example_declares_the_two_required_keys():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    body = [
        line for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "GROQ_API_KEY=" in body, "GROQ_API_KEY must be a real (empty) key"
    assert "GROQ_MODEL=" in body, "GROQ_MODEL must be a real (empty) key"


def test_env_example_contains_no_secret():
    """The template must never carry a value that looks like a live key."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("GROQ_API_KEY"):
            value = line.partition("=")[2].strip()
            assert value == "", f"template must ship an empty key, found {value!r}"


def test_env_example_lists_the_other_overrides():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in ("GROQ_TEMPERATURE", "GROQ_MAX_TOKENS", "TOP_K",
                "DISTANCE_THRESHOLD", "CHUNK_WORDS", "CHUNK_OVERLAP"):
        assert key in text, f"{key} missing from .env.example"


# --- .gitignore -------------------------------------------------------------

def test_gitignore_excludes_env():
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in rules
    assert ".env.*" in rules
    assert "! .env.example".replace(" ", "") in [r.strip() for r in rules]


@pytest.mark.parametrize("path", [".env", ".env.local", ".env.production"])
def test_env_variants_are_ignored(path, tmp_path):
    assert _is_ignored(path, tmp_path) is True


def test_env_example_is_not_ignored(tmp_path):
    """The negation matters: without it the template would be ignored too."""
    assert _is_ignored(".env.example", tmp_path) is False


def test_env_is_never_staged(tmp_path):
    """A key in .env must not reach git, whatever else is in the tree."""
    _sandbox(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=sk-should-never-be-committed\n")
    (tmp_path / "config.py").write_text("# source\n")
    staged = subprocess.run(
        ["git", "add", "-A"], cwd=tmp_path, capture_output=True, text=True
    )
    assert staged.returncode == 0, staged.stderr
    names = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.split()
    assert ".env" not in names
    assert "config.py" in names
    assert "sk-should-never-be-committed" not in subprocess.run(
        ["git", "grep", "--cached", "-l", "sk-should-never-be-committed"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout


def _sandbox(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(
        (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )


def _is_ignored(path: str, tmp_path: Path) -> bool:
    _sandbox(tmp_path)
    (tmp_path / path).write_text("x=1\n", encoding="utf-8")
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=tmp_path, capture_output=True
    )
    return result.returncode == 0


# --- python-dotenv loading --------------------------------------------------

def test_dotenv_loads_a_key_value_file(tmp_path, monkeypatch):
    """The library is actually wired up, not merely imported."""
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_MODEL=some-other-model\n", encoding="utf-8")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert load_dotenv(env_file) is True
    import os

    assert os.environ["GROQ_MODEL"] == "some-other-model"


def test_empty_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "")
    assert config._env_str("GROQ_MODEL", "fallback") == "fallback"
    monkeypatch.setenv("GROQ_MODEL", "   ")
    assert config._env_str("GROQ_MODEL", "fallback") == "fallback"
    monkeypatch.setenv("GROQ_MODEL", " explicit ")
    assert config._env_str("GROQ_MODEL", "fallback") == "explicit"


def test_shipped_empty_template_yields_usable_config(monkeypatch):
    """An empty .env (copied from the template, no values) is a valid config."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert config._env_str("GROQ_API_KEY", "") == ""
    assert config._env_str("GROQ_MODEL", "llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"


# --- missing-key behaviour --------------------------------------------------

def test_missing_key_raises_a_friendly_error(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(config.MissingConfigError) as exc:
        config.require_groq_key()
    message = str(exc.value)
    assert "GROQ_API_KEY" in message
    assert "console.groq.com" in message
    assert "cp .env.example .env" in message


def test_present_key_is_returned(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "a-key")
    assert config.require_groq_key() == "a-key"


# --- settings and allowlist -------------------------------------------------

def test_config_exposes_the_two_groq_settings():
    assert isinstance(config.GROQ_MODEL, str) and config.GROQ_MODEL
    assert isinstance(config.GROQ_TEMPERATURE, float)
    assert isinstance(config.GROQ_MAX_TOKENS, int)


def test_allowlist_is_exactly_five_schemes():
    assert len(config.SOURCES) == 5
    assert len(config.ALLOWED_URLS) == 5
    assert len(config.ALLOWED_SCHEME_CODES) == 5
    assert all(url.startswith("https://groww.in/mutual-funds/") for url in config.ALLOWED_URLS)


def test_print_settings_runs(capsys):
    config.print_settings()
    out = capsys.readouterr().out
    for source in config.SOURCES:
        assert source["url"] in out
    assert "Groq API key" in out


# --- Streamlit Cloud secrets -------------------------------------------------
# Cloud stores secrets in `st.secrets`, not os.environ. config.py mirrors them
# so `_env_str` finds them; these pin the three rules that must not drift.
#
# A plain dict stands in for `st.secrets`: the bridge only iterates it and
# subscripts it, so this avoids depending on Streamlit's private internals.


@pytest.fixture
def cloud_secret(monkeypatch):
    """Simulate a Streamlit script run with one secret configured."""
    import streamlit
    import streamlit.runtime.scriptrunner as sr

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(sr, "get_script_run_ctx", lambda: object())
    monkeypatch.setattr(streamlit, "secrets", {"GROQ_API_KEY": "gsk_from_cloud"})
    return "gsk_from_cloud"


def test_streamlit_secret_reaches_the_environment(cloud_secret):
    config._load_streamlit_secrets()
    assert os.environ["GROQ_API_KEY"] == cloud_secret


def test_streamlit_secret_does_not_override_a_real_env_var(cloud_secret):
    """.env and the platform environment must keep winning over a secret."""
    os.environ["GROQ_API_KEY"] = "gsk_from_env"
    try:
        config._load_streamlit_secrets()
        assert os.environ["GROQ_API_KEY"] == "gsk_from_env"
    finally:
        del os.environ["GROQ_API_KEY"]


def test_secrets_are_ignored_outside_a_streamlit_run(monkeypatch):
    """The CLI and the test suite import config outside any script run."""
    import streamlit
    import streamlit.runtime.scriptrunner as sr

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(sr, "get_script_run_ctx", lambda: None)
    monkeypatch.setattr(streamlit, "secrets", {"GROQ_API_KEY": "gsk_from_cloud"})
    config._load_streamlit_secrets()
    assert "GROQ_API_KEY" not in os.environ


def test_a_broken_secrets_file_does_not_break_import(monkeypatch):
    """A bad secrets.toml must surface as a missing key, not an import crash."""
    import streamlit
    import streamlit.runtime.scriptrunner as sr

    class Boom:
        def __iter__(self):
            raise RuntimeError("malformed secrets.toml")

    monkeypatch.setattr(sr, "get_script_run_ctx", lambda: object())
    monkeypatch.setattr(streamlit, "secrets", Boom())
    config._load_streamlit_secrets()
