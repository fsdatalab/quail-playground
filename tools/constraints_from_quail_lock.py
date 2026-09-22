"""Copy quail's locked versions into this project's constraints.

    python tools/constraints_from_quail_lock.py /path/to/quail/uv.lock
    uv lock

Every package that quail's lock holds at one version becomes a
``name==version`` constraint, so the playground resolves to the same
GPU stack quail tests with. Packages quail locks at several versions
(numpy, per platform) and the two git dependencies are left out.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

SKIP = {"quail-engine", "quail-b", "numpy"}
START = "constraint-dependencies = [\n"


def constraints(lock_text: str) -> list[str]:
    versions = collections.defaultdict(set)
    for match in re.finditer(
            r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock_text):
        versions[match.group(1)].add(match.group(2))
    return [f"{name}=={next(iter(found))}"
            for name, found in sorted(versions.items())
            if len(found) == 1 and name not in SKIP]


def replace_block(pyproject: str, pins: list[str]) -> str:
    start = pyproject.index(START) + len(START)
    end = pyproject.index("]\n", start)
    body = "".join(f'    "{pin}",\n' for pin in pins)
    return pyproject[:start] + body + pyproject[end:]


def main(argv=None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        raise SystemExit(__doc__)
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pins = constraints(Path(args[0]).read_text("utf-8"))
    pyproject.write_text(replace_block(pyproject.read_text("utf-8"), pins),
                         "utf-8")
    print(f"{len(pins)} constraints written to {pyproject}")


if __name__ == "__main__":
    main()
