#!/usr/bin/env python3
"""Strip transient Jupyter notebook outputs for Git clean filters."""

from __future__ import annotations

import json
import sys


def main() -> int:
    notebook = json.load(sys.stdin)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    metadata = notebook.get("metadata", {})
    metadata.pop("widgets", None)

    json.dump(notebook, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
