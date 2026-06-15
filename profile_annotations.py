"""
5/21/2026
This program looks at the CSV files and reports what it finds. 

Usage:
  python profile_annotations.py /path/to/folder/with/csvs/

  # Or target specific files:
  python profile_annotations.py file1.csv file2.csv file3.csv

  # Save report to a file to share:
  python profile_annotations.py /path/to/csvs/ > format_report.txt
"""

import csv
import re
import sys
from pathlib import Path

#helper functions 

'''This function reads the first 4096 bytes of the file (just the beginning, not the whole thing) 
and counts how many semicolons vs commas appear. 
Whichever is more common is probably the separator.'''
def sniff_separator(path: Path) -> str:
    raw = path.read_bytes()[:4096].decode("utf-8", errors="replace")
    return ";" if raw.count(";") > raw.count(",") else ","

'''
Reads data into one dictionary per row: the keys are the column names (key value pairs)
 So a row like R1, goose, 100.5, 200.0 becomes {"#": "R1", "Name": "goose", "Start": "100.5", "End": "200.0"}
'''
def read_csv_safe(path: Path, sep: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        for row in reader:
            rows.append(row)
    return rows

'''
since different files had different time formats, we need to return a description of each format
No colons, just a number → float_seconds (e.g. 138.121)
One colon → M:SS_dot (e.g. 2:56.000)
Two colons, third part has 3 digits → H:MM:SSS_dot (3-digit seconds bug) (e.g. 0:03:007.498)
Two colons normally → H:MM:SS_dot (e.g. 1:04:46.000)
Four colons → unknown
'''
def classify_time(val: str) -> str:
    val = val.strip()
    if not val:
        return "empty"
    try:
        float(val)
        return "float_seconds"
    except ValueError:
        pass
    n = val.count(":")
    has_dot = "." in val
    if n == 1:
        return f"M:SS{'_dot' if has_dot else ''}"
    if n == 2:
        # check for 3-digit seconds field like 007.498
        parts = val.split(":")
        if re.match(r"\d{3}", parts[2]):
            return "H:MM:SSS_dot (3-digit seconds bug)"
        return f"H:MM:SS{'_dot' if has_dot else ''}"
    return f"unknown ({val[:20]!r})"

#returns a dictionary of findings 
def profile_file(path: Path) -> dict:
    sep = sniff_separator(path) #find seperator
    try:
        rows = read_csv_safe(path, sep) #read the file
    except Exception as e:
        return {"path": path.name, "error": str(e)}

    if not rows:
        return {"path": path.name, "error": "no rows after header"}

    columns = list(rows[0].keys()) #get column names

    # identify time columns
    time_formats = {}
    for tcol in ["Start", "End", "Ende", "Ende "]: #It checks for all possible names for time columns (some files use "Ende" with a trailing space — a quirk we found in Bettina's files
        if tcol in columns:
            sample_vals = [r[tcol].strip() for r in rows[:10] if r[tcol].strip()]
            if sample_vals:
                fmts = {classify_time(v) for v in sample_vals}
                time_formats[tcol] = sorted(fmts)

    # label column detection
    label_col = "Name" if "Name" in columns else columns[1] if len(columns) > 1 else "?"
    extra_label_cols = [c for c in columns if c.startswith("Unnamed")] #for any extra unnamed columns with additional label tokens 

    # sample label values
    sample_labels = []
    for r in rows[:8]:
        label = r.get("Name", "").strip()
        extras = [r.get(c, "").strip() for c in extra_label_cols if r.get(c, "").strip()]
        full = " | ".join([label] + extras) if extras else label
        if full:
            sample_labels.append(full)

    # row count excluding blanks
    non_blank = [r for r in rows if any(v.strip() for v in r.values())]

    #packaging all info for main
    return {
    "path":             path.name,       # just the filename, not full path
    "separator":        repr(sep),       # ',' or ';' 
    "n_rows":           len(non_blank),  # how many real rows
    "columns":          columns,         # list of column names
    "time_formats":     time_formats,    # dict of {column: [format names]}
    "label_col":        label_col,       # which column has labels
    "extra_label_cols": extra_label_cols,# any unnamed extra label columns
    "sample_labels":    sample_labels,   # first 8 labels as examples
    "error":            None,            # None means no error occurred
    }

#Takes whatever you typed on the command line and returns a flat list of CSV file paths
def collect_paths(args: list[str]) -> list[Path]:
    paths = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.csv")))
        elif p.suffix.lower() == ".csv":
            paths.append(p)
        else:
            print(f"  [skip] not a CSV or directory: {a}", file=sys.stderr)
    return paths


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = collect_paths(sys.argv[1:])
    if not paths:
        print("No CSV files found.")
        sys.exit(1)

    print(f"Found {len(paths)} CSV file(s)\n")
    print("=" * 70)

    profiles = [profile_file(p) for p in paths]
#tells us exactly how many different cases the parser needed to handle.
    for p in profiles:
        print(f"\nFILE: {p['path']}")
        if p.get("error"):
            print(f"  ERROR: {p['error']}")
            continue
        print(f"  Rows:      {p['n_rows']}")
        print(f"  Separator: {p['separator']}")
        print(f"  Columns:   {p['columns']}")
        if p["extra_label_cols"]:
            print(f"  Extra label cols: {p['extra_label_cols']}")
        print(f"  Time formats:")
        for col, fmts in p["time_formats"].items():
            print(f"    {col}: {fmts}")
        print(f"  Sample labels:")
        for lbl in p["sample_labels"]:
            print(f"    {lbl!r}")

    #summary of unique column schemas 
    print("\n" + "=" * 70)
    print("COLUMN SCHEMA SUMMARY (unique combinations)\n")
    schemas: dict[str, list[str]] = {}
    for p in profiles:
        if p.get("error"):
            continue
        key = str(p["columns"])
        schemas.setdefault(key, []).append(p["path"])
    for schema, files in schemas.items():
        print(f"  Schema: {schema}")
        print(f"  Files ({len(files)}): {', '.join(files[:5])}")
        if len(files) > 5:
            print(f"           ... and {len(files)-5} more")
        print()

    #summary of unique time formats 
    print("TIME FORMAT SUMMARY (unique combinations)\n")
    tfmts: dict[str, list[str]] = {}
    for p in profiles:
        if p.get("error"):
            continue
        key = str(p["time_formats"])
        tfmts.setdefault(key, []).append(p["path"])
    for fmt, files in tfmts.items():
        print(f"  Format: {fmt}")
        print(f"  Files ({len(files)}): {', '.join(files[:5])}")
        if len(files) > 5:
            print(f"           ... and {len(files)-5} more")
        print()

    print("=" * 70)
    print("Done. Share this output to get a parser built for all your formats.")

if __name__ == "__main__":
    main()