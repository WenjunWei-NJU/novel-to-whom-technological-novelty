"""Export the 29 frozen submission table transcriptions as CSVs and notes."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE_IDS = [str(i) for i in range(1, 7)] + [f"S{i}" for i in range(1, 24)]


def export_tables(source, output):
    tables = json.loads(source.read_text(encoding="utf-8"))
    if list(tables) != TABLE_IDS:
        raise ValueError("The snapshot must contain Tables 1–6 and S1–S23 in order.")
    output.mkdir(parents=True, exist_ok=True)
    index = ["# Frozen submission table data\n",
             "The CSVs retain the submitted display strings and column positions. "
             "The source JSON also retains merged-cell metadata.\n"]
    for table_id, table in tables.items():
        filename = f"Table_{table_id}.csv"
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for row in table["rows"]:
                cells = []
                for cell in row:
                    cells.extend([cell["text"]] + [""] * (cell["columns"] - 1))
                writer.writerow(cells)
        index.extend([f"## [{table['caption']}]({filename})\n", table["note"] + "\n"])
    (output / "README.md").write_text("\n".join(index), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data" / "table_inputs" / "submission_tables.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Required frozen table input does not exist: {args.input}")
    export_tables(args.input, args.output)


if __name__ == "__main__":
    main()
