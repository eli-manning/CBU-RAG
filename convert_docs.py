"""
Document conversion for the CBU knowledge base.

Converts source documents that ingest.py handles poorly (or not at all) into
plain text shaped for retrieval, written to an output directory that can then
be passed to `ingest.py --dir`.

    python convert_docs.py --src "/path/to/CSDS LLM RAG Resources" --out ./converted

.docx is converted via pandoc when available (falls back to python-docx).
.xlsx four-year plans are parsed as semester grids so each course becomes a
sentence carrying its program, year and term -- raw cell dumps do not embed
usefully against natural-language questions.
"""

import argparse
import re
import subprocess
from pathlib import Path

SEMESTER_RE = re.compile(r"^(FALL|SPRING|SUMMER|WINTER)\s+\d{4}$", re.I)
YEAR_RE = re.compile(r"^YEAR\s+\d+", re.I)
COURSE_GROUP_COLS = (2, 6, 10)  # B, F, J -- the plan grid; O+ is W/F tracking


def _title_from(path: Path) -> str:
    return path.stem.replace("_", " ").replace("  ", " ").strip()


def convert_docx(src: Path, out_dir: Path) -> Path | None:
    dest = out_dir / f"{src.stem}.txt"
    try:
        subprocess.run(
            ["pandoc", str(src), "-t", "plain", "--wrap=none", "-o", str(dest)],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            import docx
        except ImportError:
            print(f"  ! skip {src.name}: no pandoc and no python-docx")
            return None
        document = docx.Document(str(src))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        dest.write_text("\n".join(parts))
    header = f"{_title_from(src)}\n\n"
    dest.write_text(header + dest.read_text(errors="ignore"))
    print(f"  docx -> {dest.name}")
    return dest


def _cell(row, col):
    if col - 1 >= len(row):
        return ""
    v = row[col - 1]
    return "" if v is None else str(v).strip()


def convert_xlsx(src: Path, out_dir: Path) -> Path | None:
    try:
        import openpyxl
    except ImportError:
        print(f"  ! skip {src.name}: openpyxl not installed")
        return None

    wb = openpyxl.load_workbook(str(src), data_only=True)
    program = _title_from(src)
    lines = [program, ""]

    for sheet in wb.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        # Group courses by (year, term) so each semester becomes its own block.
        # Chunking splits on blank lines, so this keeps a term's courses together
        # and stops unrelated semesters sharing a chunk.
        groups: dict[tuple[str, str], list[str]] = {}
        order: list[tuple[str, str]] = []
        year, terms = "", {}
        for row in rows:
            first = _cell(row, 2)
            if YEAR_RE.match(first):
                year = first
                continue
            found = {c: _cell(row, c) for c in COURSE_GROUP_COLS
                     if SEMESTER_RE.match(_cell(row, c))}
            if found:
                terms = found
                continue
            if _cell(row, 3).lower() == "course":  # header row
                continue
            for col, term in terms.items():
                code, name = _cell(row, col), _cell(row, col + 1)
                if not code or not name or "totaling" in code.lower():
                    continue
                units, status = _cell(row, col + 2), _cell(row, col + 3)
                detail = f"{code} - {name}"
                if units:
                    detail += f" ({units} units)"
                if status:
                    detail += f" [{status}]"
                key = (year, term)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(detail)

        for key in order:
            year_label, term = key
            header = f"{program} - {year_label}, {term}" if year_label else f"{program} - {term}"
            lines.append(header)
            lines.append(
                f"Courses taken in {term}"
                + (f" ({year_label})" if year_label else "")
                + f" of the {program}:"
            )
            for detail in groups[key]:
                lines.append(f"  {detail}")
            lines.append("")

    if len(lines) <= 2:
        print(f"  ! {src.name}: no course grid found, skipping")
        return None

    dest = out_dir / f"{src.stem}.txt"
    dest.write_text("\n".join(lines))
    print(f"  xlsx -> {dest.name} ({len(lines) - 2} lines)")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source directory")
    ap.add_argument("--out", default="./converted", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(Path(args.src).rglob("*")):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix == ".docx":
            convert_docx(path, out_dir)
        elif path.suffix == ".xlsx":
            convert_xlsx(path, out_dir)

    print(f"\nConverted files in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
