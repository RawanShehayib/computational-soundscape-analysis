'''
5/21/2026
Parse the annotations for the PANNs inference code.
reads both comma- and semicolon-separated files
extracts start, end, and joined label text
converts times to seconds
normalizes labels into a small set of canonical classes like bird_call, goose, speech, airplane, vehicle, frog, water, bell

Output columns:
  source_file     : filename the row came from
  region_id       : original marker/region ID
  start_s         : start time in seconds (float)
  end_s           : end time in seconds (float)
  duration_s      : duration in seconds (float, 0 for point markers)
  raw_label       : label text as written by annotator
  canonical_class : normalised class label
  uncertain       : True if mapping needs manual review

Usage format:
  python parse_annotations.py --input_dir /path/to/csvs --output annotations_clean.csv

clean annotations format note:
canonical_class: The standardised category we mapped it to using keyword rules
'''

'''
Libraries explained and pip install needed:
argparse — handles the --input_dir and --output arguments you type in the terminal
re — regular expressions, used for extracting dates from filenames and tokenising label text
sys — for sys.exit() when something goes wrong
pathlib.Path — file path handling, same as profiler
pandas — Reads CSVs into DataFrames. This is an external library (pip install pandas). 
'''
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

LABEL_RULES = [
    ("airplane",    {"airplane", "aiplane", "plane", "turbine", "turbines"}),
    ("helicopter",  {"helicopter", "helicopters", "paraglider", "paragliders"}),
    ("vehicle",     {"vehicle", "road", "engine", "enginge", "enginevehicle",
                     "truck", "car", "bus", "motor", "motors", "traffic",
                     "honking", "industrial", "machine", "train",
                     "sawing", "accelerating"}),
    ("goose",       {"goose", "gooses", "geese", "gooses", "squawk",
                     "squawking", "caw", "caaw", "scaw", "chatter",
                     "snattering", "swarm", "quacking", "quackin",
                     "qucking", "duck", "swan"}),
    ("frog",        {"frog", "frogs", "croak", "croaking", "quaking",
                     "ribbit", "queaking"}),
    ("bird_call",   {"bird", "birds", "birdcall", "birdcalls", "birdsong",
                     "wingbeat", "wings", "fluttering", "chirp", "chirping",
                     "gaggle", "swooshing", "whistle", "whistling",
                     "screech", "screeching", "srceeching", "squeak",
                     "suqeak", "ssueaking", "squeaking", "squeeking",
                     "moaning", "hooting", "hammering", "crow", "cawing",
                     "dove", "cuckoo", "cricket", "geese", "reed",
                     "singing", "cooing", "squaking", "chatting",
                     "flapping", "tweet", "tit", "raspy"}),
    ("speech",      {"human", "humans", "voice", "voices", "talking",
                     "footsteps", "footstep", "steps", "laughing",
                     "coughing", "kid", "kids", "child", "children",
                     "screaming", "female", "male", "person", "people",
                     "peopel", "recordist", "speech", "recording"}),
    
    ("water",       {"water", "splashing", "splash", "drip", "drop",
                     "dropping", "moving", "lapping", "swimming",
                     "bubble", "bubbling"}),
    ("bell",        {"bell", "bells", "church", "burch", "ringing", "ting"}),
    ("bang",        {"bang", "bangs", "boom", "echoing", "reverb", "knock",
                     "slam", "clap", "claps", "clapping"}),
    ("wind",        {"wind", "windy", "windscape", "gust"}),
    ("rustle",      {"rustling", "rustle", "branches", "rummaging",
                     "cracking", "crackling", "wooden", "wood", "leaves",
                     "grass", "gras", "reed", "rattling", "handling",
                     "wheelchair", "movements", "movement"}),
    ("click",       {"clicking", "clicks", "photocamera", "camera",
                     "metal", "crunch"}),
    ("dog",         {"dog", "barking"}),
    ("insect",      {"insect", "fly", "buzz", "buzzing", "cricket"}),
]

UNCERTAIN_TOKENS = {"subtle", "muffed", "muffeld", "distant", "background"}
DROP_NAMES = {"description", "introduction", "intro", "end", "start",
              "render", "speech data recording", "data speech recording",
              "data recording speech", "speech recording data",
              "claps", "clap"}


def parse_time(s: str) -> float | None:
    #convert any timestamp string to seconds. Returns None if unparseable.
    if s is None:
        return None
    s = str(s).strip()
    if not s or s == "-":
        return None

    # Already a float
    try:
        return float(s)
    except ValueError:
        pass

    parts = s.split(":")

    if len(parts) == 2:
        # M:SS.mmm
        try:
            return float(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return None

    if len(parts) == 3:
        # H:MM:SS[.mmm] or H:MM:SSS.mmm
        try:
            h = float(parts[0])
            m = float(parts[1])
            sec = float(parts[2])   # handles both SS and SSS
            return h * 3600 + m * 60 + sec
        except ValueError:
            return None

    if len(parts) == 4:
        # SMPTE H:MM:SS:FF — assume 30fps
        try:
            h   = float(parts[0])
            m   = float(parts[1])
            sec = float(parts[2])
            frm = float(parts[3])
            return h * 3600 + m * 60 + sec + frm / 30.0
        except ValueError:
            return None

    return None

# Column detection helpers
def get_id_col(cols: list[str]) -> str:
    for c in cols:
        if c.strip().lstrip("\ufeff") in ("#", "Marker"):
            return c
    return cols[0]

def get_end_col(cols: list[str]) -> str | None:
    for c in cols:
        if c.strip() in ("End", "Ende", "Ende "):
            return c
    return None

def get_extra_label_cols(cols: list[str]) -> list[str]:
    """Return unnamed/extra columns that may carry additional label tokens."""
    return [c for c in cols if c.strip() == "" or c.startswith("Unnamed")]

# Label normalisation
def build_raw_label(row: pd.Series, extra_cols: list[str]) -> str:
    name = str(row.get("Name", "")).strip()
    extras = [str(row[c]).strip() for c in extra_cols
              if c in row.index and pd.notna(row[c]) and str(row[c]).strip()
              and str(row[c]).strip().lower() not in ("nan", "")]
    parts = [name] + extras
    return " ".join(p for p in parts if p).lower()


def classify(raw: str) -> tuple[str, bool]:
    tokens = set(re.findall(r"[a-z]+", raw.lower()))
    uncertain = bool(tokens & UNCERTAIN_TOKENS)
    for class_name, keywords in LABEL_RULES:
        if tokens & keywords:
            return class_name, uncertain
    return "unknown", True


def is_junk(raw: str) -> bool:
    stripped = raw.strip().lower()
    if stripped in DROP_NAMES:
        return True
    # Catch variants like "speech data recording" anywhere in label
    for drop in DROP_NAMES:
        if stripped == drop:
            return True
    return False

# Single-file parser
def parse_file(path: Path) -> pd.DataFrame:
    # Detect separator
    raw_bytes = path.read_bytes()[:4096].decode("utf-8", errors="replace")
    sep = ";" if raw_bytes.count(";") > raw_bytes.count(",") else ","

    try:
        df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8",
                         encoding_errors="replace")
    except Exception as e:
        print(f"  [ERROR reading {path.name}] {e}")
        return pd.DataFrame()

    # Normalise column names (strip BOM, trailing spaces)
    df.columns = [c.encode("utf-8").decode("utf-8-sig").strip()
                  for c in df.columns]
    # Re-strip after BOM removal
    df.columns = [c.strip() for c in df.columns]

    cols        = df.columns.tolist()
    id_col      = get_id_col(cols)
    end_col     = get_end_col(cols)
    extra_cols  = get_extra_label_cols(cols)
    has_end     = end_col is not None

    records = []
    skipped = 0

    for _, row in df.iterrows():
        region_id = str(row.get(id_col, "")).strip()
        raw_label = build_raw_label(row, extra_cols)

        # Drop empty or junk rows
        if not raw_label or is_junk(raw_label):
            skipped += 1
            continue

        start_s = parse_time(row.get("Start"))
        if start_s is None:
            skipped += 1
            continue

        if has_end:
            end_s = parse_time(row.get(end_col))
        else:
            end_s = None  # point marker file

        # If end is missing/unparseable, treat as point event
        if end_s is None or end_s <= start_s:
            end_s = start_s

        duration_s = round(end_s - start_s, 3)
        canonical, uncertain = classify(raw_label)

        records.append({
            "source_file":     path.name,
            "region_id":       region_id,
            "start_s":         round(start_s, 3),
            "end_s":           round(end_s, 3),
            "duration_s":      duration_s,
            "raw_label":       raw_label,
            "canonical_class": canonical,
            "uncertain":       uncertain,
        })

    result = pd.DataFrame(records)
    print(f"  {path.name:<65} {len(result):>4} rows  ({skipped} skipped)")
    return result



# File selection — prefer (edited) over plain for same date
def select_files(folder: Path) -> list[Path]:
    all_csv = sorted(folder.rglob("*.csv"))  # rglob searches all subfolders

    edited  = {p for p in all_csv if "(edited)" in p.name or "edited" in p.stem.lower().split("_")[0]}
    plain   = {p for p in all_csv if p not in edited}

    # For each plain file, check if an edited version covers the same date
    # by looking for matching date string (first 8 digits)
    date_re = re.compile(r"(\d{8})")

    def extract_date(p: Path) -> str | None:
        m = date_re.search(p.stem)
        return m.group(1) if m else None

    edited_dates = {extract_date(p) for p in edited} - {None}

    selected = list(edited)
    for p in plain:
        d = extract_date(p)
        if d and d in edited_dates:
            pass  # skip — edited version exists for this date
        else:
            selected.append(p)

    return sorted(selected)

# Main
def main():
    parser = argparse.ArgumentParser(description="Parse all Vogelinsel annotation CSVs")
    parser.add_argument("--input_dir", "-i", required=True,
                        help="Folder containing annotation CSV files")
    parser.add_argument("--output", "-o", default="annotations_clean.csv",
                        help="Output CSV path (default: annotations_clean.csv)")
    parser.add_argument("--all_files", action="store_true",
                        help="Parse ALL files including duplicates (default: prefer edited)")
    args = parser.parse_args()

    folder = Path(args.input_dir)
    if not folder.exists():
        print(f"ERROR: folder not found: {folder}")
        sys.exit(1)

    if args.all_files:
        files = sorted(folder.rglob("*.csv"))
    else:
        files = select_files(folder)

    print(f"\nParsing {len(files)} file(s) from: {folder}\n")

    all_frames = []
    for f in files:
        result = parse_file(f)
        if not result.empty:
            all_frames.append(result)

    if not all_frames:
        print("\nNo data parsed.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)

    print(f"\n{'='*65}")
    print(f"  Total annotations : {len(combined)}")
    print(f"  Files parsed      : {len(all_frames)}")
    print(f"\n  Class distribution:")
    for cls, n in combined["canonical_class"].value_counts().items():
        bar = "█" * (n // 20)
        print(f"    {cls:<15} {n:>5}  {bar}")
    uncertain_n = combined["uncertain"].sum()
    print(f"\n  Uncertain mappings: {uncertain_n}  ({100*uncertain_n/len(combined):.1f}%)")
    print(f"{'='*65}\n")

    combined.to_csv(args.output, index=False)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
