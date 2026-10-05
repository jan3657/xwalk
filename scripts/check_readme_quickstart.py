#!/usr/bin/env python
"""Run the README quickstart exactly as written, against a given installation.

    python scripts/check_readme_quickstart.py --bin /path/to/fresh-venv/bin [--workdir DIR]

The README marks its quickstart with `<!-- quickstart:begin -->` and
`<!-- quickstart:end -->`. Between them, every ```bash line is a command and the ```python
block is the `quickstart.py` the text tells the reader to save. Commands run in order in
an empty working directory, with `--bin` first on PATH, so `xwalk` and `python` are the
installation under test (a clean venv with the built wheel in CI and in a release check),
not the source tree. Any non-zero exit fails the check.

No network and no credentials: the quickstart is the offline route.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
BEGIN, END = "<!-- quickstart:begin -->", "<!-- quickstart:end -->"
FENCE = re.compile(r"^```(\w+)\n(.*?)^```", re.S | re.M)
SCRIPT_NAME = "quickstart.py"


@dataclass(frozen=True)
class Step:
    """One thing the reader does: run a command, or save the Python block."""

    command: tuple[str, ...] = ()
    script: str = ""


def quickstart_steps(readme: str) -> list[Step]:
    """The quickstart section of `readme` as an ordered list of steps."""
    start, end = readme.find(BEGIN), readme.find(END)
    if start < 0 or end < start:
        raise ValueError(f"README has no {BEGIN} ... {END} section")
    steps: list[Step] = []
    for language, body in FENCE.findall(readme[start:end]):
        if language == "python":
            steps.append(Step(script=body))
        elif language == "bash":
            for line in body.splitlines():
                words = shlex.split(line, comments=True)
                if words:
                    steps.append(Step(command=tuple(words)))
    if not any(s.script for s in steps) or not any(s.command for s in steps):
        raise ValueError("the quickstart section needs a bash block and a python block")
    return steps


def run(steps: list[Step], bin_dir: Path, workdir: Path) -> int:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
    env.pop("PYTHONPATH", None)  # the source tree must not leak into the clean install
    for step in steps:
        if step.script:
            (workdir / SCRIPT_NAME).write_text(step.script, encoding="utf-8")
            print(f"$ (saved {SCRIPT_NAME}, {len(step.script.splitlines())} lines)")
            continue
        print(f"$ {shlex.join(step.command)}", flush=True)
        done = subprocess.run(step.command, cwd=workdir, env=env, check=False)
        if done.returncode != 0:
            print(f"FAILED: exit {done.returncode}: {shlex.join(step.command)}", file=sys.stderr)
            return 1
    print(f"README quickstart OK in {workdir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--bin", required=True, type=Path, help="bin/Scripts dir of the venv")
    parser.add_argument("--workdir", type=Path, help="empty directory to run in (default: temp)")
    args = parser.parse_args(argv)
    steps = quickstart_steps(README.read_text(encoding="utf-8"))
    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        if any(args.workdir.iterdir()):
            parser.error(f"{args.workdir} is not empty")
        return run(steps, args.bin.resolve(), args.workdir.resolve())
    with tempfile.TemporaryDirectory(prefix="xwalk-quickstart-") as tmp:
        return run(steps, args.bin.resolve(), Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
