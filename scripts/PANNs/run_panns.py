"""
Usage:
    python run_panns.py --audio "path/to/file.flac" --output predictions_20200923

Output:
    predictions_20200923.npz  — compressed array, load with np.load()
    predictions_20200923_summary.csv — top classes per 10s window (human-readable)
"""


'''
libraries:
librosa : loads the .flac audio file and converts it to the right format (32000 Hz, mono)
numpy : handles the large arrays of numbers, saves the .npz file
pandas : creates the human-readable summary CSV
pathlib.Path : file path handling
argparse : handles command line arguments
panns_inference : provides SoundEventDetection and labels
'''
import argparse
import sys
import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path


def run_sed(audio_path: str, output_stem: str, device: str = 'cuda',
            chunk_sec: int = 60):
    """
    Run SoundEventDetection on a long audio file in chunks.

    PANNs SoundEventDetection expects short clips (the model was trained
    on 10s clips). For long recordings we process in 60s chunks and
    concatenate the framewise output.

    Loads the entire flac file into memory as a numpy array of numbers representing the sound wave
    
    Args:
        audio_path  : path to .flac file
        output_stem : output file path without extension
        device      : 'cuda' or 'cpu'
        chunk_sec   : process this many seconds at a time (keep ≤120s)
    """
    from panns_inference import SoundEventDetection, labels

    sr = 32000 #sample rate
    print(f"\nLoading: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    total_sec = len(audio) / sr
    print(f"  Duration: {total_sec/60:.1f} min  ({len(audio):,} samples @ {sr} Hz)")

    print(f"\nLoading PANNs SoundEventDetection model (device={device})...")
    #Loads the pretrained CNN14 model weights
    sed = SoundEventDetection(
        checkpoint_path=None,
        device=device,
        interpolate_mode='nearest'
    )
    print("  Model ready.")

    chunk_samples = chunk_sec * sr
    n_chunks = int(np.ceil(len(audio) / chunk_samples)) #rounds up so the last chunk is always included even if it's shorter than 60 seconds
    all_framewise = []

    print(f"\nProcessing {n_chunks} chunks of {chunk_sec}s each...")
    for i in range(n_chunks):
        start = i * chunk_samples
        end   = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        #If it's the last chunk and shorter than 60 seconds, pads it with zeros to make it full length — the model requires a fixed size input        if len(chunk) < chunk_samples:
        
        
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        '''
        Each row is one moment in time — every 10 milliseconds of the recording. So row 1 = the first 10ms, row 2 = the next 10ms, row 100 = second 1 of the recording, row 6000 = minute 1, and so on all the way to the end of the 2-hour recording.
        Each column is one of PANNs' 527 sound categories — "Speech", "Goose", "Dog", "Airplane", "Water", "Crow", etc.
        Each cell contains a number between 0 and 1 — PANNs' confidence that that sound was present at that moment.
        '''
        chunk_batch = chunk[None, :]           # (1, samples)
        framewise = sed.inference(chunk_batch) # (1, time_steps, 527)
        all_framewise.append(framewise[0])     # (time_steps, 527)

        pct = 100 * (i + 1) / n_chunks
        elapsed_min = (i + 1) * chunk_sec / 60
        print(f"  [{pct:5.1f}%]  chunk {i+1}/{n_chunks}  "
              f"({elapsed_min:.1f} min of audio processed)")

    #  Concatenate and trim to actual audio length 
    framewise_full = np.concatenate(all_framewise, axis=0)  # (total_frames, 527)

    # Each frame = hop_size/sr seconds = 320/32000 = 0.01s
    hop_sec = 320 / 32000
    expected_frames = int(np.ceil(total_sec / hop_sec))
    framewise_full = framewise_full[:expected_frames]

    print(f"\nFramewise output shape: {framewise_full.shape}")
    print(f"  ({framewise_full.shape[0]} frames × {framewise_full.shape[1]} classes)")
    print(f"  Time resolution: {hop_sec*1000:.0f} ms per frame")

    '''Save compressed numpy file .npz for comparisons
    The .npz file contains PANNs' complete "annotation" of the recording — for every 10ms moment across the whole 2 hours, a probability for each of 527 sound classes
    '''
    npz_path = output_stem + '.npz'
    np.savez_compressed(
        npz_path,
        framewise_output=framewise_full,
        labels=np.array(labels),
        hop_sec=hop_sec,
        total_sec=total_sec
    )
    print(f"\nSaved → {npz_path}")

    #  Save human-readable summary (top classes per 10s window) 
    window_frames = int(10 / hop_sec)   # frames per 10s
    n_windows = int(np.ceil(framewise_full.shape[0] / window_frames))

    rows = []
    for w in range(n_windows):
        s = w * window_frames
        e = min(s + window_frames, framewise_full.shape[0])
        window_probs = framewise_full[s:e].max(axis=0)  # max prob in window
        top5_idx = np.argsort(window_probs)[::-1][:5]
        rows.append({
            'start_s': round(w * 10, 1),
            'end_s':   round(min((w + 1) * 10, total_sec), 1),
            'top1_class': labels[top5_idx[0]],
            'top1_prob':  round(float(window_probs[top5_idx[0]]), 3),
            'top2_class': labels[top5_idx[1]],
            'top2_prob':  round(float(window_probs[top5_idx[1]]), 3),
            'top3_class': labels[top5_idx[2]],
            'top3_prob':  round(float(window_probs[top5_idx[2]]), 3),
        })

    summary_path = output_stem + '_summary.csv'
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved → {summary_path}")

    # Print overall top classes
    print("\nTop 15 classes detected overall (by max probability):")
    overall = framewise_full.max(axis=0)
    top_idx = np.argsort(overall)[::-1][:15]
    for idx in top_idx:
        print(f"  {overall[idx]:.3f}  {labels[idx]}")

''' 
Handles the command line arguments and checks the audio file exists before calling run_sed(). 
The four arguments are --audio (path to flac), --output (where to save), --device (cuda or cpu), and --chunk_sec (how big each chunk is)
'''
def main():
    parser = argparse.ArgumentParser(
        description="Run PANNs SoundEventDetection on a .flac file"
    )
    parser.add_argument("--audio",  "-a", required=True,
                        help="Path to the .flac audio file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output file stem (no extension), e.g. predictions_20200923")
    parser.add_argument("--device", "-d", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--chunk_sec", type=int, default=60,
                        help="Chunk size in seconds (default: 60)")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"ERROR: audio file not found: {args.audio}")
        sys.exit(1)

    run_sed(
        audio_path=args.audio,
        output_stem=args.output,
        device=args.device,
        chunk_sec=args.chunk_sec
    )


if __name__ == "__main__":
    main()