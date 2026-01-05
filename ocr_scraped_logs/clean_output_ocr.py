from pathlib import Path
import csv
import os

target_dir = Path("./output/output_unified_table_ocr_fast")
output_dir = Path("./output/cleaned_output_unified_table_ocr")
i = 0


def is_line_broken(row: list[str]) -> bool:
    if not row:
        return False
    last = row[-1].strip()
    if last != "Oo":
        return False
    return all(cell.strip() == "" for cell in row[:-1])


for csv_path in target_dir.rglob("*csv"):
    i += 1
    print(f"Reading: {csv_path}")

    rel = csv_path.relative_to(target_dir)
    out_path = output_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
    with csv_path.open("r", newline="", encoding="utf-8")as fin, \
            tmp_path.open("w", newline="", encoding="utf-8")as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader, None)
        if header is None:
            writer.writerow(header)
        for row in reader:
            if is_line_broken(row):
                print("found broken line")
                continue
            writer.writerow(row)
    tmp_path.replace(out_path)
print(i)
