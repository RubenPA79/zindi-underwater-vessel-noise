"""Build the submission-ready DOCX from ``entrega/WRITEUP.md``.

A deliberately small Markdown subset: ATX headings, paragraphs, bullet and
numbered lists, pipe tables, fenced code blocks, images and inline ``**bold**`` /
``code``. That is everything the write-up uses, and nothing that would silently
produce a wrong document.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.build_writeup
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

PROJECT_DIR = Path(__file__).resolve().parents[1]
ENTREGA = PROJECT_DIR / "entrega"
SOURCE = ENTREGA / "WRITEUP.md"
OUTPUT = ENTREGA / "WRITEUP.docx"

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
BLOCK_START = re.compile(r"^(\||- |\d+\.\s|#|!\[|```)")


def add_inline(paragraph, text: str) -> None:
    """Add text to a paragraph, honouring **bold**, *italic* and `code` spans."""
    for chunk in INLINE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            run = paragraph.add_run(chunk[2:-2])
            run.bold = True
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            run = paragraph.add_run(chunk[1:-1])
            run.italic = True
        elif chunk.startswith("`") and chunk.endswith("`"):
            run = paragraph.add_run(chunk[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        else:
            paragraph.add_run(chunk)


def add_table(document: Document, rows: list[str]) -> None:
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    cells = [row for row in cells if not all(set(cell) <= set("-: ") for cell in row)]
    if not cells:
        return
    table = document.add_table(rows=len(cells), cols=len(cells[0]))
    table.style = "Light Grid Accent 1"
    for row_index, row in enumerate(cells):
        for column_index, value in enumerate(row):
            paragraph = table.cell(row_index, column_index).paragraphs[0]
            add_inline(paragraph, value)
            if row_index == 0:
                for run in paragraph.runs:
                    run.bold = True


def build() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            paragraph = document.add_paragraph()
            run = paragraph.add_run("\n".join(block))
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
            index += 1
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                path = (ENTREGA / match.group(2)).resolve()
                if path.exists():
                    document.add_picture(str(path), width=None)
                    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    caption = document.add_paragraph(match.group(1))
                    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    caption.runs[0].italic = True
                    caption.runs[0].font.size = Pt(8.5)
            index += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = document.add_heading(level=min(level, 4))
            add_inline(heading, stripped[level:].strip())
            index += 1
            continue

        if stripped.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            add_table(document, block)
            document.add_paragraph()
            continue

        if stripped.startswith("- ") or re.match(r"^\d+\.\s", stripped):
            ordered = not stripped.startswith("- ")
            pattern = r"^\d+\.\s" if ordered else r"^- "
            style = "List Number" if ordered else "List Bullet"
            while index < len(lines) and re.match(pattern, lines[index].strip()):
                content = [re.sub(pattern, "", lines[index].strip())]
                index += 1
                while index < len(lines) and lines[index].strip() and not BLOCK_START.match(lines[index].strip()):
                    content.append(lines[index].strip())
                    index += 1
                paragraph = document.add_paragraph(style=style)
                add_inline(paragraph, " ".join(content))
            continue

        if not stripped:
            index += 1
            continue

        paragraph_lines = []
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^(\||- |\d+\.\s|#|!\[|```)", lines[index].strip()
        ):
            paragraph_lines.append(lines[index].strip())
            index += 1
        paragraph = document.add_paragraph()
        add_inline(paragraph, " ".join(paragraph_lines))

    document.save(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(PROJECT_DIR)}")
    print(f"paragraphs: {len(document.paragraphs)}, tables: {len(document.tables)}")


if __name__ == "__main__":
    build()
