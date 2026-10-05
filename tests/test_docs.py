"""The documentation is checkable, so it gets checked.

Prose cannot be unit-tested, but three things about it can: that its links resolve, that
its Python examples are at least syntactically real, and that no page is orphaned. All
three rot silently otherwise -- a broken link in a reference page is invisible until a
reader hits it.

`docs/superpowers/` holds design plans and specs, not user documentation. It is excluded:
those are historical records and their links point at things that moved.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# [text](target) -- not images, and not reference-style definitions.
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
# Reference pages display signatures with no body -- `def open(...) -> BM25Retriever`, or
# `Cls(\n  arg: T = default,\n)`. That is annotated pseudo-code, not an expression.
SIGNATURE_OPENER = re.compile(r"^(def |async def |class |@|\)|\.\.\.|[A-Za-z_][\w.]*\()")


def _is_signature_display(block: str) -> bool:
    """True when every top-level line opens a signature. Continuation lines are indented,
    so only column-zero lines are examined."""
    top_level = [line for line in block.splitlines() if line.strip() and not line[0].isspace()]
    return bool(top_level) and all(SIGNATURE_OPENER.match(line) for line in top_level)


def _pages() -> list[Path]:
    pages = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "CHANGELOG.md"]
    pages += [p for p in DOCS.rglob("*.md") if "superpowers" not in p.parts]
    return sorted(p for p in pages if p.exists())


PAGES = _pages()
IDS = [str(p.relative_to(ROOT)) for p in PAGES]


def test_the_documentation_tree_exists():
    """A guard on the guard: if the glob silently matched nothing, everything below
    would pass by vacuum."""
    names = {p.name for p in PAGES}
    assert {"README.md", "concepts.md", "components.md"} <= names
    assert len(PAGES) >= 15


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_every_relative_link_resolves(page: Path):
    broken = []
    for target in LINK.findall(page.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        if not path:
            continue
        if not (page.parent / path).resolve().exists():
            broken.append(target)
    assert not broken, f"{page.relative_to(ROOT)} links to missing paths: {broken}"


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_every_python_example_parses(page: Path):
    """A snippet that does not parse cannot be one a reader can copy. This does not
    prove an example runs -- it proves nobody mangled it."""
    for index, block in enumerate(FENCE.findall(page.read_text(encoding="utf-8"))):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            if _is_signature_display(block):
                continue
            pytest.fail(f"{page.relative_to(ROOT)} python block {index}: {exc}\n{block}")


def test_the_docs_index_links_to_every_page():
    """An unlinked page is a page nobody finds."""
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    orphans = [
        p.relative_to(DOCS).as_posix()
        for p in DOCS.rglob("*.md")
        if "superpowers" not in p.parts
        and "claude-upgrade" not in p.parts
        and p.name != "README.md"
        and p.relative_to(DOCS).as_posix() not in index
    ]
    assert not orphans, f"docs/README.md does not link to: {orphans}"


def test_the_readme_points_at_the_documentation():
    assert "docs/README.md" in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_no_page_advertises_a_symbol_the_library_does_not_export(page: Path):
    """Catches the reference page that documents a function someone later renamed."""
    import xwalk.batch
    import xwalk.evaluate
    import xwalk.llm
    import xwalk.ops
    import xwalk.prompts.contract
    import xwalk.retrieval.bm25
    import xwalk.review

    modules = {
        "xwalk.evaluate": xwalk.evaluate,
        "xwalk.review": xwalk.review,
        "xwalk.batch": xwalk.batch,
        "xwalk.llm": xwalk.llm,
        "xwalk.ops": xwalk.ops,
        "xwalk.prompts.contract": xwalk.prompts.contract,
        "xwalk.retrieval.bm25": xwalk.retrieval.bm25,
    }
    text = page.read_text(encoding="utf-8")
    missing = []
    for name, module in modules.items():
        for symbol in re.findall(rf"from {re.escape(name)} import ([\w, ]+)", text):
            for part in (s.strip() for s in symbol.split(",")):
                if part and not hasattr(module, part):
                    missing.append(f"{name}.{part}")
    assert not missing, f"{page.relative_to(ROOT)} imports names that do not exist: {missing}"


def _quickstart_checker():
    import importlib.util
    import sys

    path = ROOT / "scripts" / "check_readme_quickstart.py"
    spec = importlib.util.spec_from_file_location("check_readme_quickstart", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_the_readme_quickstart_is_offline_and_complete():
    """The quickstart is the first thing a reader runs; it must need no credentials."""
    steps = _quickstart_checker().quickstart_steps((ROOT / "README.md").read_text("utf-8"))
    commands = [s.command for s in steps if s.command]
    assert commands[0][:2] == ("xwalk", "init")
    assert ("python", "quickstart.py") in commands
    assert not any("match" in c for c in commands), "`xwalk match` needs a real endpoint"
    validate = next(c for c in commands if c[:2] == ("xwalk", "validate"))
    assert "--no-credentials" in validate


def test_the_readme_quickstart_runs_exactly_as_written(tmp_path):
    """Runs every quickstart command against this environment's `xwalk`. CI and the
    release workflow run the same script against a clean install of the built wheel."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_readme_quickstart.py")]
        + ["--bin", str(Path(sys.executable).parent), "--workdir", str(tmp_path / "w")],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    # The README quotes this line; keep the two in step.
    quoted = "complete {'matched': 3, 'unmatched': 1, 'total': 4} model calls: 7"
    assert quoted in done.stdout
    assert quoted in " ".join((ROOT / "README.md").read_text("utf-8").split())
    assert (tmp_path / "w" / "demo" / "reviewed.csv").exists()
