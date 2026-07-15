"""Tests for src/gen_module_docs.py (AST inventory, import graph, MODULES.md check)."""
from pathlib import Path

import pytest

from src import gen_module_docs as g


def _fixture(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "models").mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "models" / "__init__.py").write_text("")
    (src / "a.py").write_text("def foo():\n    return 1\n\ndef _priv():\n    return 2\n")
    (src / "models" / "b.py").write_text(
        "from src.a import foo\nimport src.a\n\ndef bar():\n    return foo()\n")
    return src


def test_build_inventory(tmp_path, monkeypatch) -> None:
    src = _fixture(tmp_path)
    monkeypatch.setattr(g, "_SRC", src)
    inv = g.build_inventory(src)
    assert inv["src.a"] == ["foo", "_priv"]
    assert inv["src.models.b"] == ["bar"]
    assert "src.__init__" not in inv          # __init__.py skipped


def test_build_import_graph(tmp_path, monkeypatch) -> None:
    src = _fixture(tmp_path)
    monkeypatch.setattr(g, "_SRC", src)
    graph = g.build_import_graph(src)
    assert graph["src.models.b"] == ["src.a"]   # edge to the imported module, deduped
    assert graph["src.a"] == []                 # no intra-src imports


def test_render_and_generate(tmp_path, monkeypatch) -> None:
    src = _fixture(tmp_path)
    monkeypatch.setattr(g, "_SRC", src)
    out = tmp_path / "module_diagram.md"
    g.generate(out)
    text = out.read_text()
    assert "Module Diagram & Inventory" in text
    assert "src.models.b" in text and "`bar`" in text
    assert "└─▶ src.a" in text


def test_check_modules_md_detects_missing(tmp_path, monkeypatch) -> None:
    src = _fixture(tmp_path)
    monkeypatch.setattr(g, "_SRC", src)
    md = tmp_path / "MODULES.md"
    # bar: full entry (anchor + fenced pseudocode). foo: anchor but NO pseudocode block.
    md.write_text(
        "# Modules\n\n"
        "## src.models.b\n\n"
        "### `bar`\n`() -> int`\n\nCall foo.\n\n```\nreturn foo()\n```\n\n"
        "## src.a\n\n"
        "### `foo`\n`() -> int`\n\nReturn 1 (no pseudocode block).\n"
    )
    missing = g.check_modules_md(md)
    assert "src.a.foo" in missing                     # anchor present but no pseudocode fence
    assert "src.models.b.bar" not in missing          # anchor + fenced pseudocode → documented
    # private helpers are exempt
    assert not any(m.endswith("._priv") for m in missing)


def test_modules_md_complete() -> None:
    """The real MODULES.md documents every public src function with pseudocode (rule 6)."""
    if not g._MODULES.exists():
        pytest.skip("docs/MODULES.md not present")
    missing = g.check_modules_md()
    assert missing == [], (
        "MODULES.md is missing a pseudocode entry for: " + ", ".join(missing)
    )
