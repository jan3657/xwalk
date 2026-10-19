"""Path confinement for the MCP server (`xwalk.confine`), without the MCP SDK.

Every path an MCP tool receives, and every file a job it names refers to, must resolve
(symlinks followed) inside one of the server's roots. These tests pin the confinement
function itself; `tests/test_mcp.py` checks the tools refuse through it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xwalk import ops
from xwalk.confine import PathRefused, Roots


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "root").mkdir()
    return tmp_path / "root"


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.yaml").write_text("name: secret\n", encoding="utf-8")
    return tmp_path / "outside"


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:  # pragma: no cover - Windows without symlink rights
        pytest.skip("cannot create symlinks here")


# --- the roots themselves ----------------------------------------------------------------


def test_the_default_root_is_the_working_directory(root, monkeypatch):
    monkeypatch.chdir(root)
    roots = Roots()
    assert roots.paths == (root.resolve(),)
    assert roots.resolve("job.yaml", what="job") == root.resolve() / "job.yaml"


def test_a_root_must_be_an_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        Roots([tmp_path / "missing"])
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        Roots([tmp_path / "file"])


# --- tool arguments ----------------------------------------------------------------------


def test_relative_and_absolute_paths_inside_a_root_resolve(root, monkeypatch):
    monkeypatch.chdir(root)
    (root / "data").mkdir()
    roots = Roots([root])
    assert roots.resolve("data/job.yaml", what="job") == root.resolve() / "data" / "job.yaml"
    assert roots.resolve(str(root / "run"), what="out") == root.resolve() / "run"
    assert roots.resolve(".", what="out") == root.resolve()  # the root itself is inside
    # a path that does not exist yet (an output directory) is fine
    assert roots.resolve("data/new/run", what="out") == root.resolve() / "data/new/run"


def test_dotdot_traversal_is_refused(root, outside, monkeypatch):
    monkeypatch.chdir(root)
    roots = Roots([root])
    with pytest.raises(PathRefused, match="outside the allowed roots") as refused:
        roots.resolve("../outside/secret.yaml", what="job")
    assert refused.value.code == "path_outside_roots"
    assert "job" in str(refused.value)
    with pytest.raises(PathRefused):
        roots.resolve("sub/../../outside", what="out")


def test_an_absolute_path_outside_every_root_is_refused(root, outside):
    roots = Roots([root])
    with pytest.raises(PathRefused):
        roots.resolve(str(outside / "secret.yaml"), what="job")
    with pytest.raises(PathRefused):
        roots.resolve("/", what="out")


def test_a_sibling_sharing_the_roots_name_prefix_is_outside(tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj-evil").mkdir()
    roots = Roots([tmp_path / "proj"])
    with pytest.raises(PathRefused):
        roots.resolve(str(tmp_path / "proj-evil" / "job.yaml"), what="job")


def test_a_symlink_pointing_out_of_a_root_is_refused(root, outside):
    _symlink(root / "escape", outside)
    _symlink(root / "secret.yaml", outside / "secret.yaml")
    roots = Roots([root])
    with pytest.raises(PathRefused):
        roots.resolve(str(root / "escape" / "secret.yaml"), what="job")
    with pytest.raises(PathRefused):
        roots.resolve(str(root / "secret.yaml"), what="job")
    # writing through the link would land outside too, even where nothing exists yet
    with pytest.raises(PathRefused):
        roots.resolve(str(root / "escape" / "new-run"), what="out")


def test_a_symlink_that_stays_inside_is_followed(root):
    (root / "real").mkdir()
    _symlink(root / "alias", root / "real")
    assert Roots([root]).resolve(str(root / "alias" / "x"), what="out") == (
        root.resolve() / "real" / "x"
    )


def test_any_of_several_roots_will_do(tmp_path, outside):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    roots = Roots([first, second])
    assert roots.resolve(str(second / "run"), what="out") == second.resolve() / "run"
    with pytest.raises(PathRefused) as refused:
        roots.resolve(str(outside), what="out")
    assert str(first.resolve()) in str(refused.value) and str(second.resolve()) in str(
        refused.value
    )


def test_an_unusable_path_is_refused_not_crashed(root):
    with pytest.raises(PathRefused):
        Roots([root]).resolve("bad\0name", what="job")


# --- what a job file refers to ------------------------------------------------------------


@pytest.fixture
def job(root: Path) -> Path:
    ops.init(root / "quickstart")
    return root / "quickstart" / "job.yaml"


def _edit(job: Path, old: str, new: str) -> None:
    text = job.read_text(encoding="utf-8")
    assert old in text
    job.write_text(text.replace(old, new), encoding="utf-8")


def test_a_job_whose_files_are_inside_passes(root, job):
    Roots([root]).check_job(job)  # does not raise


def test_a_job_referring_to_data_outside_is_refused(root, outside, job):
    (outside / "targets.csv").write_text("id,label\nX,x\n", encoding="utf-8")
    _edit(job, "path: targets.csv", f"path: {outside / 'targets.csv'}")
    with pytest.raises(PathRefused, match=r"target\.path") as refused:
        Roots([root]).check_job(job)
    assert refused.value.code == "path_outside_roots"


def test_a_job_climbing_out_with_dotdot_is_refused(root, job):
    _edit(job, "path: sources.csv", "path: ../../../outside/sources.csv")
    with pytest.raises(PathRefused, match=r"source\.path"):
        Roots([root]).check_job(job)


def test_a_job_whose_prompt_slots_are_outside_is_refused(root, outside, job):
    _edit(job, "slots: slots.yaml", f"slots: {outside / 'slots.yaml'}")
    with pytest.raises(PathRefused, match=r"prompts\.slots"):
        Roots([root]).check_job(job)


def test_a_job_whose_data_file_is_a_symlink_out_is_refused(root, outside, job):
    (outside / "targets.csv").write_text("id,label\nX,x\n", encoding="utf-8")
    (job.parent / "targets.csv").unlink()
    _symlink(job.parent / "targets.csv", outside / "targets.csv")
    with pytest.raises(PathRefused, match=r"target\.path"):
        Roots([root]).check_job(job)


def test_a_dense_model_given_as_a_local_path_must_be_inside(root, outside, job):
    _edit(
        job,
        "retrievers:\n",
        f"retrievers:\n  - kind: dense\n    name: dense\n    model: {outside}\n",
    )
    with pytest.raises(PathRefused, match=r"retrievers\.0\.model"):
        Roots([root]).check_job(job)


def test_a_dense_model_named_by_hub_id_is_not_a_path(root, job):
    _edit(
        job,
        "retrievers:\n",
        "retrievers:\n  - kind: dense\n    name: dense\n"
        "    model: sentence-transformers/all-MiniLM-L6-v2\n",
    )
    Roots([root]).check_job(job)


def _sql_job(job: Path, url: str) -> None:
    text = job.read_text(encoding="utf-8")
    text = text.replace(
        "source:\n  kind: csv\n  path: sources.csv\n",
        f"source:\n  kind: sql\n  url: {url!r}\n  query: select 1\n",
    )
    job.write_text(text, encoding="utf-8")


def test_a_sqlite_database_outside_is_refused(root, outside, job, monkeypatch):
    monkeypatch.chdir(root)
    _sql_job(job, f"sqlite:///{outside / 'db.sqlite'}")
    with pytest.raises(PathRefused, match=r"source\.url"):
        Roots([root]).check_job(job)


def test_a_sqlite_database_inside_passes(root, job, monkeypatch):
    monkeypatch.chdir(root)
    _sql_job(job, f"sqlite:///{root / 'db.sqlite'}")
    Roots([root]).check_job(job)


def test_a_relative_sqlite_path_resolves_against_the_working_directory(root, job, monkeypatch):
    _sql_job(job, "sqlite:///db.sqlite")  # SQLAlchemy opens it relative to the process
    monkeypatch.chdir(root)
    Roots([root]).check_job(job)
    monkeypatch.chdir(root.parent)
    with pytest.raises(PathRefused, match=r"source\.url"):
        Roots([root]).check_job(job)


def test_a_network_database_is_refused(root, job):
    _sql_job(job, "postgresql://user@db.internal/prod")
    with pytest.raises(PathRefused, match="network database"):
        Roots([root]).check_job(job)


def test_a_cluster_job_referring_to_data_outside_is_refused(root, outside):
    job = root / "cluster.yaml"
    job.write_text(
        "kind: cluster\nname: c\n"
        f"source: {{kind: csv, path: {outside / 'labels.csv'}}}\n"
        "templates: {query: '{{ label }}'}\n"
        "llm: {kind: openai_compat, model: m, base_url: 'https://x.invalid/v1'}\n",
        encoding="utf-8",
    )
    with pytest.raises(PathRefused, match=r"source\.path"):
        Roots([root]).check_job(job)


def test_an_invalid_job_is_left_to_validation(root):
    job = root / "broken.yaml"
    job.write_text("name: [unclosed\n", encoding="utf-8")
    Roots([root]).check_job(job)  # nothing to check; ops reports the invalid job


# --- directories written or read beneath a confined path -----------------------------------


def test_a_symlink_beneath_a_directory_argument_is_refused(root, outside):
    index = root / "index"
    index.mkdir()
    _symlink(index / "bm25", outside)  # writes to index/bm25/... would land outside
    roots = Roots([root])
    with pytest.raises(PathRefused, match=r"index_dir.*bm25"):
        roots.resolve(str(index), what="index_dir", tree=True)
    nested = root / "run" / "index"
    nested.mkdir(parents=True)
    _symlink(nested / "ledger.sqlite", outside / "secret.yaml")
    with pytest.raises(PathRefused, match="ledger.sqlite"):
        roots.resolve(str(root / "run"), what="out", tree=True)


def test_a_symlink_beneath_that_stays_inside_is_accepted(root):
    (root / "real").mkdir()
    (root / "run").mkdir()
    _symlink(root / "run" / "alias", root / "real")
    roots = Roots([root])
    assert roots.resolve(str(root / "run"), what="out", tree=True) == root.resolve() / "run"


def test_a_differently_cased_path_to_the_same_root_is_accepted(root, outside):
    upper = Path(str(root.parent / root.name.upper()))
    if not upper.exists() or not upper.samefile(root):
        pytest.skip("case-sensitive filesystem")
    roots = Roots([root])
    assert roots.resolve(str(upper / "job.yaml"), what="job").name == "job.yaml"
    assert roots.resolve(str(upper / "new" / "run"), what="out").name == "run"
    other = Path(str(outside.parent / outside.name.upper()))
    with pytest.raises(PathRefused):
        roots.resolve(str(other / "secret.yaml"), what="job")
