"""
run_clap.py
===========
Runs LAION CLAP zero-shot sound detection on a .flac file.

Slides a window across the full recording and for each window.
CLAP outputs similarity scores for your 15 custom class descriptions.

Usage:
    python run_clap.py \
        --audio   "G:\\path\\to\\file.flac" \
        --output  clap_predictions_20200923.csv \
        --window  5
"""

import argparse
import sys
import numpy as np
import pandas as pd
import librosa

'''
CLAP's text encoder converts the class descriptions into 512-number vectors in the same embedding space. 
Then cosine similarity tells you which description is closest to the audio.
Same categories from parse annotations
Uses the CLAP text encoder for cosine similarity: it hears the audio -> compares it with the class descriptions and 
thats where the audio vs text vector comparison happens
listen to (window) 5 seconds → compare against 15 classes → pick the closest one
'''
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

CLAP_SR = 48000 #48 khz SR 


def load_audio(audio_path: str) -> tuple:
    '''
    len(audio)/sr/60 converts from number of samples to minutes, divide by samples per second to get seconds, divide by 60 to get minutes
    '''
    print(f"\nLoading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=CLAP_SR, mono=True)
    print(f"  Duration: {len(audio)/sr/60:.1f} min  |  Sample rate: {sr} Hz")
    return audio, sr


def cosine_sim_matrix(audio_emb: np.ndarray,
                      text_embs: np.ndarray) -> np.ndarray:
    '''Compute cosine similarity between one audio embedding and all text embeddings.
    Takes two inputs:

        audio_emb — one audio vector, shape (512,)
        text_embs — all 15 text vectors, shape (15, 512)

    Returns 15 similarity scores, shape (15,)
    '''
    audio_norm = audio_emb / (np.linalg.norm(audio_emb) + 1e-8)
    text_norms = text_embs / (
        #np.linalg.norm computes the length of the vector a
        #the 1e-8 to avoid dividing by 0 for small numbers 
        #axis=1 to normalize each row (each text vector) separately.
        np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8) 
    #the @ operator is for matrix multiplication: dot product between each text vector and audio vector
    #gives 15 cosine similarities in one operation
    return text_norms @ audio_norm  # (n_classes,)


def main():
    parser = argparse.ArgumentParser(
        description="Run LAION CLAP zero-shot detection on a .flac file"
    )
    parser.add_argument("--audio",  "-a", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--window", "-w", type=int, default=5,
                        help="Window size in seconds (default: 5)")
    args = parser.parse_args()

    audio, sr = load_audio(args.audio)
    total_sec = len(audio) / sr

    print(f"\nLoading LAION CLAP model...")
    try:
        import laion_clap
    except ImportError:
        print("ERROR: pip install laion-clap")
        sys.exit(1)

    model = laion_clap.CLAP_Module(enable_fusion=False, device='cpu')
    model.load_ckpt()
    print("  CLAP model ready.")

    print("\nEmbedding class descriptions...")
    classes      = list(CLASS_DESCRIPTIONS.keys())
    descriptions = list(CLASS_DESCRIPTIONS.values())
    #is a matrix of shape (15, 512) — 15 rows (one per class), 512 columns (the embedding numbers)
    text_embs    = model.get_text_embedding(descriptions, use_tensor=False)
    print(f"  Embedded {len(classes)} classes.")
    for cls, desc in CLASS_DESCRIPTIONS.items():
        print(f"    {cls:<12} → '{desc}'")

    window_samples = args.window * sr #number of audio samples in one window. For 5 seconds at 48kHz: 5 × 48000 = 240,000 samples
    n_windows      = int(np.ceil(len(audio) / window_samples)) #for rounding up the last short window

    print(f"\nProcessing {n_windows} windows of {args.window}s each...")
    print(f"  Total recording: {total_sec/60:.1f} minutes\n")

    rows = []

    for w in range(n_windows):
        '''
        start_s: start time in seconds (0, 5, 10, 15...)
        end_s: end time in seconds (5, 10, 15, 20...)
        s: start sample index (0, 240000, 480000...)
        e: end sample index (240000, 480000, 720000...)
        segment: the actual audio data for this window
        '''
        start_s = w * args.window
        end_s   = min((w + 1) * args.window, total_sec)

        s       = w * window_samples
        e       = min(s + window_samples, len(audio))
        segment = audio[s:e]

        # Pad if shorter than window
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)))

        try:
            # Pass numpy array directly — no file needed
            audio_data = segment.reshape(1, -1)  # (1, samples)
            audio_emb  = model.get_audio_embedding_from_data(
                x=audio_data, use_tensor=False)[0]  # (512,)

            # Cosine similarity against all class descriptions
            sims      = cosine_sim_matrix(audio_emb, text_embs)
            top_idx   = int(np.argmax(sims))
            top_class = classes[top_idx]
            top_sim   = float(sims[top_idx])

            row = {
                'start_s':   round(start_s, 1),
                'end_s':     round(end_s, 1),
                'top_class': top_class,
                'top_sim':   round(top_sim, 3),
            }
            for cls, sim in zip(classes, sims):
                row[f'sim_{cls}'] = round(float(sim), 3)

            rows.append(row)

        except Exception as e:
            print(f"  [WARNING] window {w+1} failed: {e}")
            continue

        if (w + 1) % 50 == 0 or (w + 1) == n_windows:
            pct = 100 * (w + 1) / n_windows
            print(f"  [{pct:5.1f}%]  window {w+1}/{n_windows}  "
                  f"({start_s/60:.1f} min)  → {top_class} ({top_sim:.3f})")

    #  Save predictions
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")
    print(f"Shape: {df.shape[0]} windows × {df.shape[1]} columns")

    print("\nOverall detections (top class per window):")
    counts = df['top_class'].value_counts()
    for cls, n in counts.items():
        pct = 100 * n / len(df)
        bar = '█' * int(pct / 2)
        print(f"  {cls:<12} {n:>5} windows  ({pct:4.1f}%)  {bar}")


if __name__ == "__main__":
    main()