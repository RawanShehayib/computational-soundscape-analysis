"""
compare_clap.py
Compares human annotations against audio using LAION CLAP.
how similar is this audio to what the human wrote
listen to annotation interval → compare against 15 classes → measure similarity w/ annotations

For each annotation interval:
  1. Extracts the actual audio segment
  2. Gets CLAP audio embedding
  3. Gets CLAP text embedding for raw label ("muffled raspy goose caw")
  4. Gets CLAP text embedding for canonical description ("goose or duck calling...")
  5. Computes cosine similarity for both

This produces the same table format as compare_semantic.py (PANNs):
  Class | N | Mean sim (raw) | Mean sim (canonical) | Match% (raw) | Match% (can) | Det | Miss

Usage:
    python compare_clap.py \
        --annotations annotations_clean.csv \
        --audio       "G:\\path\\to\\file.flac" \
        --source_file 20200923 \
        --output      comparison_clap_20200923.csv

    # First 30 minutes only:
    python compare_clap.py \
        --annotations annotations_clean.csv \
        --audio       "G:\\path\\to\\file.flac" \
        --source_file 20200923 \
        --output      comparison_clap_20200923_30min.csv \
        --max_s       1800
"""

import argparse
import sys
import numpy as np
import pandas as pd
import librosa

SIM_THRESHOLD = 0.2

MAX_DURATION  = 15.0
CLAP_SR       = 48000

CLASS_DESCRIPTIONS = {
    'goose':      'goose or duck calling, quacking, honking waterfowl',
    'bird_call':  'bird singing, chirping, tweeting, bird call or song',
    'speech':     'human speech, talking, voice, conversation, footsteps',
    'vehicle':    'car engine, truck, road traffic, motor vehicle noise',
    'airplane':   'airplane flying overhead, aircraft engine, jet noise',
    'helicopter': 'helicopter rotor noise, helicopter flying overhead',
    'frog':       'frog croaking, toad calling, amphibian sound',
    'water':      'water splashing, stream, rain, dripping water',
    'bell':       'church bell ringing, bell chiming',
    'bang':       'loud bang, explosion, gunshot, impact noise',
    'wind':       'wind blowing, gusty wind noise',
    'rustle':     'rustling leaves, rustling grass, rustling branches',
    'click':      'clicking sound, camera click, mechanical click',
    'dog':        'dog barking, dog growling',
    'insect':     'insect buzzing, cricket chirping, fly buzzing',
}

ALL_CLASSES = list(CLASS_DESCRIPTIONS.keys())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def extract_segment(audio, sr, start_s, end_s, max_dur=MAX_DURATION):
    duration = end_s - start_s
    if duration > max_dur:
        mid     = (start_s + end_s) / 2
        start_s = mid - max_dur / 2
        end_s   = mid + max_dur / 2
    s = max(0, int(start_s * sr))
    e = min(len(audio), int(end_s * sr))
    return audio[s:e]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", "-a", required=True)
    parser.add_argument("--audio",       "-A", required=True)
    parser.add_argument("--source_file", "-s", required=True)
    parser.add_argument("--output",      "-o", default="comparison_clap.csv")
    # ── CHANGE 1 — added --max_s argument ────────────────────────────────────
    # Remove or don't pass this argument to run on the full recording
    parser.add_argument("--max_s", type=float, default=None,
                        help="Only include annotations before this time in seconds (e.g. 1800 for 30 min)")
    # ─────────────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    print(f"Loading annotations: {args.annotations}")
    ann_all = pd.read_csv(args.annotations)
    ann = ann_all[ann_all['source_file'].str.contains(
        args.source_file, case=False, na=False)]
    ann = ann[ann['canonical_class'] != 'unknown']
    # ── CHANGE 2 — optional time filter ──────────────────────────────────────
    # Remove this block to run on the full recording
    if args.max_s is not None:
        ann = ann[ann['start_s'] < args.max_s]
        print(f"  Filtered to first {args.max_s/60:.0f} minutes")
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  Annotations: {len(ann)}")

    print(f"\nLoading audio: {args.audio}")
    audio, sr = librosa.load(args.audio, sr=CLAP_SR, mono=True)
    print(f"  Duration: {len(audio)/sr/60:.1f} min")

    print(f"\nLoading LAION CLAP model...")
    try:
        import laion_clap
    except ImportError:
        print("ERROR: pip install laion-clap")
        sys.exit(1)

    model = laion_clap.CLAP_Module(enable_fusion=False, device='cpu')
    model.load_ckpt()
    print("  CLAP model ready.")

    print("\nPre-computing text embeddings...")
    can_embs = {}
    for cls, desc in CLASS_DESCRIPTIONS.items():
        emb = model.get_text_embedding([desc], use_tensor=False)
        can_embs[cls] = emb[0]

    unique_raw = ann['raw_label'].unique().tolist()
    raw_embs = {}
    batch_size = 32
    for i in range(0, len(unique_raw), batch_size):
        batch = unique_raw[i:i+batch_size]
        emb_list = model.get_text_embedding(batch, use_tensor=False)
        for text, emb in zip(batch, emb_list):
            raw_embs[text] = emb
        print(f"  Raw labels embedded: {min(i+batch_size, len(unique_raw))}/{len(unique_raw)}")
    print(f"  Done. Embedded {len(CLASS_DESCRIPTIONS)} canonical + {len(unique_raw)} raw labels.")

    print(f"\nComparing {len(ann)} annotations...")
    results = []

    for i, (_, row) in enumerate(ann.iterrows()):
        cls = row['canonical_class']
        if cls not in CLASS_DESCRIPTIONS:
            continue

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ann)} done...")

        segment = extract_segment(audio, sr, row['start_s'], row['end_s'])
        if len(segment) < sr * 0.1:
            continue

        try:
            audio_data = segment.reshape(1, -1)
            audio_emb  = model.get_audio_embedding_from_data(
                x=audio_data, use_tensor=False)[0]

            sim_raw = cosine_sim(audio_emb, raw_embs[row['raw_label']])
            sim_can = cosine_sim(audio_emb, can_embs[cls])

            results.append({
                'region_id':       row['region_id'],
                'start_s':         row['start_s'],
                'end_s':           row['end_s'],
                'raw_label':       row['raw_label'],
                'canonical_class': cls,
                'best_sim_raw':    round(sim_raw, 3),
                'best_sim_can':    round(sim_can, 3),
                'match_raw':       sim_raw >= SIM_THRESHOLD,
                'match_can':       sim_can >= SIM_THRESHOLD,
            })

        except Exception as e:
            print(f"  ERROR on {row['region_id']}: {e}")

    if not results:
        print("\nERROR: No results produced.")
        sys.exit(1)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved per-annotation results → {args.output}")

    # ── CHANGE 3 — added Det/Miss columns to summary table ───────────────────
    # To revert: remove Det/Miss columns from print and summary_rows
    print("\nPer-class CLAP similarity summary:")
    print(f"\n  {'Class':<15} {'N':>5}  "
          f"{'Mean sim (raw)':>14}  {'Mean sim (canonical)':>20}  "
          f"{'Match%(raw)':>10}  {'Det':>4}  {'Miss':>4}  "
          f"{'Match%(can)':>10}  {'Det':>4}  {'Miss':>4}")
    print(f"  {'-'*95}")

    summary_rows = []
    for cls in sorted(ALL_CLASSES):
        sub = df[df['canonical_class'] == cls]
        if len(sub) == 0:
            summary_rows.append({
                'class': cls, 'n_annotations': 0,
                'mean_sim_raw_label': 0.0, 'mean_sim_canonical': 0.0,
                'match_pct_raw': 0.0, 'detected_raw': 0, 'missed_raw': 0,
                'match_pct_canonical': 0.0, 'detected_can': 0, 'missed_can': 0,
            })
            print(f"  {cls:<15} {'0':>5}  {'—':>14}  {'—':>20}  "
                  f"{'—':>10}  {'—':>4}  {'—':>4}  {'—':>10}  {'—':>4}  {'—':>4}")
            continue

        mean_raw      = sub['best_sim_raw'].mean()
        mean_can      = sub['best_sim_can'].mean()
        match_raw_pct = sub['match_raw'].mean() * 100
        match_can_pct = sub['match_can'].mean() * 100
        detected_raw  = round(match_raw_pct / 100 * len(sub))
        detected_can  = round(match_can_pct / 100 * len(sub))

        print(f"  {cls:<15} {len(sub):>5}  "
              f"{mean_raw:>14.3f}  {mean_can:>20.3f}  "
              f"{match_raw_pct:>10.1f}%  {detected_raw:>4}  {len(sub)-detected_raw:>4}  "
              f"{match_can_pct:>10.1f}%  {detected_can:>4}  {len(sub)-detected_can:>4}")

        summary_rows.append({
            'class':               cls,
            'n_annotations':       len(sub),
            'mean_sim_raw_label':  round(mean_raw, 3),
            'mean_sim_canonical':  round(mean_can, 3),
            'match_pct_raw':       round(match_raw_pct, 1),
            'detected_raw':        detected_raw,
            'missed_raw':          len(sub) - detected_raw,
            'match_pct_canonical': round(match_can_pct, 1),
            'detected_can':        detected_can,
            'missed_can':          len(sub) - detected_can,
        })
    # ─────────────────────────────────────────────────────────────────────────

    summary_path = args.output.replace('.csv', '_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved summary → {summary_path}")

    print("\n--- High similarity: CLAP and human agree ---")
    top = df.nlargest(5, 'best_sim_raw')[
        ['raw_label', 'best_sim_raw', 'best_sim_can', 'canonical_class']]
    print(top.to_string(index=False))

    print("\n--- Low similarity: CLAP and human disagree ---")
    bot = df.nsmallest(5, 'best_sim_raw')[
        ['raw_label', 'best_sim_raw', 'best_sim_can', 'canonical_class']]
    print(bot.to_string(index=False))


if __name__ == "__main__":
    main()