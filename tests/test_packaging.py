import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

HEAVY = (
    "torch",
    "sentence-transformers",
    "faiss",
    "rdflib",
    "sqlalchemy",
    "litellm",
    "numpy",
    "scikit-learn",
)


def _base_dependencies() -> list[str]:
    block = re.search(r"^dependencies = \[(.*?)^\]", PYPROJECT, re.S | re.M)
    assert block, "could not find the base dependencies block"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_no_heavy_dependency_is_in_the_base_install():
    base = " ".join(_base_dependencies()).lower()
    for name in HEAVY:
        assert name not in base, f"{name} must live behind an extra"


def test_every_declared_extra_exists():
    for extra in ("dense", "ontology", "sql", "litellm", "all", "dev"):
        assert f"{extra} = [" in PYPROJECT


def test_the_python_floor_is_declared():
    assert 'requires-python = ">=3.10"' in PYPROJECT


def test_the_cli_entry_point_is_declared():
    assert 'xwalk = "xwalk.cli.main:main"' in PYPROJECT


def test_the_project_urls_are_real_not_placeholders():
    """A `<owner>` placeholder shipped to PyPI is a dead link on the project page."""
    assert "[project.urls]" in PYPROJECT
    assert "<owner>" not in PYPROJECT


def test_the_licence_file_exists_and_is_referenced():
    """`license = "MIT"` with no LICENSE file is a licensing claim with nothing behind it."""
    licence = ROOT / "LICENSE"
    assert licence.exists()
    assert "MIT License" in licence.read_text(encoding="utf-8")
    assert 'license-files = ["LICENSE"]' in PYPROJECT


def test_the_changelog_documents_the_declared_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', PYPROJECT, re.M)
    assert declared
    base = declared.group(1).split(".dev")[0]
    assert f"[{base}]" in changelog, f"CHANGELOG.md has no section for {base}"


def test_a_contributor_entry_point_exists():
    assert (ROOT / "CONTRIBUTING.md").exists()


def test_the_package_never_calls_itself_crosswalk():
    hits = [p for p in (ROOT / "src").rglob("*.py") if "crosswalk" in p.read_text("utf-8").lower()]
    assert not hits, f"'crosswalk' appears in {hits}"


def test_prompt_skeletons_are_packaged():
    from xwalk.prompts.contract import BASE_DIR

    assert {p.name for p in BASE_DIR.glob("*.j2")} == {
        "select.j2",
        "score.j2",
        "verify.j2",
        "rewrite.j2",
    }


def test_py_typed_marker_is_present():
    assert (ROOT / "src" / "xwalk" / "py.typed").exists()


def test_the_cli_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "xwalk.cli", "--version"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert result.stdout.strip()


def test_the_version_is_importable_and_matches_pyproject():
    from xwalk import __version__

    declared = re.search(r'^version = "([^"]+)"', PYPROJECT, re.M)
    assert declared and declared.group(1) == __version__


def test_the_platform_document_lists_the_supported_matrix():
    text = (ROOT / "docs" / "platforms.md").read_text(encoding="utf-8")
    for entry in ("manylinux", "aarch64", "macOS", "Windows"):
        assert entry in text


def test_the_platform_document_forbids_automatic_fallback():
    text = (ROOT / "docs" / "platforms.md").read_text(encoding="utf-8").lower()
    assert "never automatic" in text or "not automatic" in text


def test_ci_checks_the_same_paths_as_the_local_gate():
    """CI that lints a narrower tree than the developer does is CI that passes while
    the local gate fails."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for path in ("src", "tests", "examples", "scripts"):
        assert f"ruff check {path}" in ci or re.search(rf"ruff check [^\n]*\b{path}\b", ci)


def test_the_version_is_not_duplicated_in_the_source():
    """Two sources of truth for a version is how a release ends up mislabelled.

    `__init__.py` may name a fallback for a source checkout, but the installed value
    must come from distribution metadata.
    """
    init = (ROOT / "src" / "xwalk" / "__init__.py").read_text(encoding="utf-8")
    assert "importlib.metadata" in init or "from importlib import metadata" in init


def test_the_installed_version_comes_from_metadata():
    from importlib.metadata import version

    from xwalk import __version__

    assert __version__ == version("xwalk")
