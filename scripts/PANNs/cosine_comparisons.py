"""
Semantically compares PANNs predictions against human annotations
using sentence transformer embeddings and cosine similarity.
Usage:
    python cosine_comparisons.py \
        --annotations annotations_clean.csv \
        --predictions  predictions_20200923.npz \
        --source_file  20200923 \
        --output       comparison_semantic_20200923.csv

Install dependency first:
    pip install sentence-transformers
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path


TOP_N_PANNS = 5 # only considers top 5 PANNs classes by probability

SIM_THRESHOLD = 0.3 # cosine similarity > 0.3, then it is a match


def load_predictions(npz_path: str) -> tuple:
    data = np.load(npz_path, allow_pickle=True)
    framewise = data['framewise_output']   
    panns_labels = list(data['labels'])    # 527  class names
    hop_sec = float(data['hop_sec']) # the bridge between seconds and frame numbers
    return framewise, panns_labels, hop_sec


def get_top_panns_classes(framewise: np.ndarray,
                           panns_labels: list,
                           start_s: float,
                           end_s: float,
                           hop_sec: float,
                           top_n: int = 5) -> list[tuple]:
    """
    First, it converts to frame indices ( If a human marked a sound from second 348 to 392, and each frame is 0.01 seconds, then start_frame = 348 / 0.01 = 34,800)
    Then, using framwise, max, and argsort it slices the window, finds max probability per class, sorts to get top 5
    """

    '''
    this is where the comparison is processed:
    1. reads the human annotation every row and converts it into frame indices by dividing by 10ms (hop_sec)
    2. panns looks for those rows 
    3. cosine similarity happens
    '''
    start_frame = max(0, int(start_s / hop_sec))
    end_frame   = min(len(framewise) - 1, int(end_s / hop_sec) + 1)

    if start_frame >= end_frame:
        return []

    window = framewise[start_frame:end_frame]       # (frames, 527)
    #finding the peak probability for every one of the 527 classes
    max_probs = window.max(axis=0)                  # (527,)
    top_idx = np.argsort(max_probs)[::-1][:top_n]

    return [(panns_labels[i], round(float(max_probs[i]), 3))
            for i in top_idx]


def build_embeddings(model, texts: list[str]) -> np.ndarray:
    '''Takes a list of text strings and converts each one into a vector of 384 numbers using the sentence transformer. 
    This vector captures the meaning of the text. Similar meanings produce similar vectors.'''
    return model.encode(texts, convert_to_numpy=True,
                        show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a) #calculates the length of the vector a
    norm_a = np.linalg.norm(a) #multiplies corresponding numbers together and sums them up — measures how much the vectors point in the same direction
    norm_b = np.linalg.norm(b)
    #note: dividing by both lengths for normalization
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    parser = argparse.ArgumentParser(
        description="Semantic comparison of PANNs vs human annotations"
    )
    
    '''reading all arguements in the command line'''
    parser.add_argument("--annotations", "-a", required=True)
    parser.add_argument("--predictions", "-p", required=True)
    parser.add_argument("--source_file", "-s", required=True,
                        help="Partial match for source_file column")
    parser.add_argument("--output", "-o", default="comparison_semantic.csv")
    parser.add_argument("--top_n", type=int, default=TOP_N_PANNS,
                        help=f"Top N PANNs classes per window (default {TOP_N_PANNS})")
    args = parser.parse_args()

    print(f"\nLoading annotations: {args.annotations}")
    '''
    Reads all 5,445 rows from annotations_clean.csv. 
    Then filters to only keep rows where the source_file column contains the file we are working on.
    Drops any rows with canonical_class = 'unknown' since those couldn't be mapped and aren't useful for comparison.
    '''
    ann_all = pd.read_csv(args.annotations)
    ann = ann_all[ann_all['source_file'].str.contains(
        args.source_file, case=False, na=False)]
    ann = ann[ann['canonical_class'] != 'unknown']
    ann = ann[ann['start_s'] < 1800]  # first 30 minutes only
    print(f"  Annotations for this file: {len(ann)}")

    # Load PANNs predictions
    print(f"Loading predictions: {args.predictions}")
    #Calls load_predictions() which opens the .npz file and returns:
    framewise, panns_labels, hop_sec = load_predictions(args.predictions)
    print(f"  Framewise shape: {framewise.shape}")

    # Load sentence transformer
    print("\nLoading sentence transformer model...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        sys.exit(1)

    # all-MiniLM-L6-v2t: converts any text into a 384-number embedding vector
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Model loaded.")

    #  Pre-embed all 527 PANNs class names 
    print("\nEmbedding all 527 PANNs class names...")
    panns_embeddings = build_embeddings(model, panns_labels)  # (527, 384)
    print("  Done.")

    #  Per-annotation comparison 
    print(f"\nComparing {len(ann)} annotations...")
    results = []

    for _, row in ann.iterrows():
        # Returns the 5 strongest PANNs detections during this annotation's time window
        top_classes = get_top_panns_classes(
            framewise, panns_labels,
            row['start_s'], row['end_s'],
            hop_sec, args.top_n
        )
        if not top_classes:
            continue

        top_names  = [c[0] for c in top_classes]
        top_probs  = [c[1] for c in top_classes]
        top_str    = ', '.join(f"{n}({p})" for n, p in top_classes)

        '''
        Embed the human raw label.
        Converts "muffled raspy goose caw" into a 384-number vector, and separately converts "goose" into another 384-number vector.
        '''
        raw_emb = build_embeddings(model, [row['raw_label']])[0]

        # Embed the canonical class name
        can_emb = build_embeddings(model, [row['canonical_class']])[0]

        # For each top PANNs class, compute similarity to both
        best_sim_raw = 0.0
        best_sim_can = 0.0
        best_match_raw = ''
        best_match_can = ''
        '''
        Starts with best similarity at 0. For each of the 5 PANNs class names, looks up its pre-computed embedding, then computes cosine similarity against both the raw label embedding and the canonical embedding.
        If the similarity is higher than the current best, update the best.
        '''
        for name in top_names:
            idx = panns_labels.index(name)
            panns_emb = panns_embeddings[idx]

            sim_raw = cosine_similarity(raw_emb, panns_emb)
            sim_can = cosine_similarity(can_emb, panns_emb)

            if sim_raw > best_sim_raw:
                best_sim_raw = sim_raw
                best_match_raw = name
            if sim_can > best_sim_can:
                best_sim_can = sim_can
                best_match_can = name

        results.append({
            'region_id':         row['region_id'],
            'start_s':           row['start_s'],
            'end_s':             row['end_s'],
            'raw_label':         row['raw_label'],
            'canonical_class':   row['canonical_class'],
            # Top PANNs detections in this window
            'top_panns_classes': top_str,
            # Similarity using raw label text
            'best_sim_raw':      round(best_sim_raw, 3),
            'best_match_raw':    best_match_raw,
            'match_raw':         best_sim_raw >= SIM_THRESHOLD,
            # Similarity using canonical class name
            'best_sim_can':      round(best_sim_can, 3),
            'best_match_can':    best_match_can,
            'match_can':         best_sim_can >= SIM_THRESHOLD,
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved per-annotation results → {args.output}")

    #  Per-class summary 
    #  Per-class summary 
    print("\nPer-class similarity summary:")
    print(f"\n  {'Class':<15} {'N':>5}  "
          f"{'Mean sim (raw)':>14}  {'Mean sim (canonical)':>20}  "
          f"{'Match%(raw)':>10}  {'Det':>4}  {'Miss':>4}  "
          f"{'Match%(can)':>10}  {'Det':>4}  {'Miss':>4}")
    print(f"  {'-'*95}")

    summary_rows = []
    for cls in sorted(df['canonical_class'].unique()):
        sub = df[df['canonical_class'] == cls]
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

    summary_path = args.output.replace('.csv', '_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved class summary → {summary_path}")

    #examples
    print("\n--- High similarity: PANNs and human agree ---")
    top = df.nlargest(5, 'best_sim_raw')[
        ['raw_label','best_match_raw','best_sim_raw','canonical_class']]
    print(top.to_string(index=False))

    print("\n--- Low similarity: PANNs and human disagree ---")
    bot = df.nsmallest(5, 'best_sim_raw')[
        ['raw_label','best_match_raw','best_sim_raw','canonical_class']]
    print(bot.to_string(index=False))

    print("\n--- Cases where raw label helps vs canonical ---")
    diff = df[abs(df['best_sim_raw'] - df['best_sim_can']) > 0.1].copy()
    diff['sim_diff'] = diff['best_sim_raw'] - diff['best_sim_can']
    diff = diff.nlargest(5, 'sim_diff')[
        ['raw_label','canonical_class',
         'best_sim_raw','best_sim_can','sim_diff']]
    if len(diff):
        print(diff.to_string(index=False))
    else:
        print("  (no large differences found)")


if __name__ == "__main__":
    main()
