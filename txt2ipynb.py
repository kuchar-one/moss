#!/usr/bin/env python3
"""
Convert MOSS_Explained.txt to MOSS_Explained.ipynb

Format of .txt file:
- Lines starting with '# MARKDOWN' begin a markdown cell (content follows)
- Lines starting with '# CODE' begin a code cell (content follows)
- Empty lines within cells are preserved
- '# END' ends the current cell

Usage: python txt2ipynb.py
"""

import json
import re


def parse_txt_to_cells(txt_path):
    """Parse the custom txt format into notebook cells."""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    cells = []
    current_type = None
    current_content = []

    for line in content.split("\n"):
        if line.strip() == "# MARKDOWN":
            if current_type and current_content:
                cells.append((current_type, "\n".join(current_content)))
            current_type = "markdown"
            current_content = []
        elif line.strip() == "# CODE":
            if current_type and current_content:
                cells.append((current_type, "\n".join(current_content)))
            current_type = "code"
            current_content = []
        elif line.strip() == "# END":
            if current_type and current_content:
                cells.append((current_type, "\n".join(current_content)))
            current_type = None
            current_content = []
        elif current_type:
            current_content.append(line)

    # Handle last cell if no END marker
    if current_type and current_content:
        cells.append((current_type, "\n".join(current_content)))

    return cells


def cells_to_notebook(cells):
    """Convert cells to Jupyter notebook format."""
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    for cell_type, content in cells:
        # Strip trailing whitespace from content but preserve internal structure
        content = content.rstrip()

        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": content.split("\n") if content else [],
        }

        # Add newlines back for proper format (except last line)
        if cell["source"]:
            cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [
                cell["source"][-1]
            ]

        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

        notebook["cells"].append(cell)

    return notebook


def main():
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, "MOSS_Explained.txt")
    ipynb_path = os.path.join(script_dir, "MOSS_Explained.ipynb")

    cells = parse_txt_to_cells(txt_path)
    notebook = cells_to_notebook(cells)

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"✅ Converted {len(cells)} cells to {ipynb_path}")


if __name__ == "__main__":
    main()
