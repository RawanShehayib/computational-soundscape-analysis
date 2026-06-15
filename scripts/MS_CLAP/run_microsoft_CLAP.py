import torch
import librosa
import pandas as pd
import numpy as np
import soundfile as sf
import tempfile
import os
from msclap import CLAP


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

#to know if it running of CPU or GPU. if it is slow, it is running on CPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

AUDIO_PATH = "data/vogelinsel_20200923_30min.wav"
ANNOTATIONS_PATH = "data/annotations_clean.csv"
RECORDING_DATE = "20200923"
MAX_DURATION_MIN = 30

def load_data(audio_path, annotations_path, recording_date, max_duration_min):
    audio, sr = librosa.load(audio_path, sr=48000, mono=True)
    
    ann_all = pd.read_csv(annotations_path)
    ann = ann_all[ann_all['source_file'].str.contains(recording_date, case=False, na=False)]
    ann = ann[ann['canonical_class'] != 'unknown']
    ann = ann[ann['start_s'] < max_duration_min * 60]
    
    print(f"Audio: {len(audio)/sr/60:.1f} min")
    print(f"Annotations: {len(ann)}")
    return audio, sr, ann

def load_models(use_cuda=True):
    clap_model = CLAP(version='2023', use_cuda=use_cuda)
    clapcap = CLAP(version='clapcap', use_cuda=use_cuda)
    print("Both models ready")
    return clap_model, clapcap

def embed_classes(clap_model, class_descriptions):
    classes = list(class_descriptions.keys())
    descriptions = list(class_descriptions.values())
    print("Embedding class descriptions...")
    class_text_embs = clap_model.get_text_embeddings(descriptions)
    print(f"Done — {len(classes)} classes embedded")
    return classes, descriptions, class_text_embs

def run_inference(audio, sr, ann, clap_model, clapcap, class_text_embs, 
                  classes, output_path, max_duration=15.0, sim_threshold=0.2):
    
    results = []
    audio_emb_store = {}
    print(f"Processing {len(ann)} annotations...")

    for i, (_, row) in enumerate(ann.iterrows()):
        cls     = row['canonical_class']
        start_s = row['start_s']
        end_s   = row['end_s']

        # Extract audio segment
        duration = end_s - start_s
        if duration > max_duration:
            mid     = (start_s + end_s) / 2
            start_s = mid - max_duration / 2
            end_s   = mid + max_duration / 2
        s = max(0, int(start_s * sr))
        e = min(len(audio), int(end_s * sr))
        segment = audio[s:e]
        if len(segment) < sr * 0.1:
            continue

        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, segment, sr)

        try:
            # Step 1 — Zero-shot classification
            audio_emb  = clap_model.get_audio_embeddings([tmp.name])
            class_sims = clap_model.compute_similarity(
                audio_emb, class_text_embs)[0].detach().cpu().numpy()
            top_idx    = int(np.argmax(class_sims))
            zs_class   = classes[top_idx]
            zs_sim     = float(class_sims[top_idx])
            zs_correct = (zs_class == cls)

            # Store for retrieval step
            audio_emb_store[row['region_id']] = {
                'emb': audio_emb, 'cls': cls,
                'raw_label': row['raw_label']
            }

            # Step 2 — Audio captioning
            caption = clapcap.generate_caption([tmp.name])[0]

            # Step 3 — Caption vs human label using CLAP text encoder
            caption_emb = clap_model.get_text_embeddings([caption])
            raw_emb     = clap_model.get_text_embeddings([row['raw_label']])
            can_emb     = clap_model.get_text_embeddings([cls])

            cap_vs_raw = float(clap_model.compute_similarity(
                caption_emb, raw_emb)[0][0].detach().cpu())
            cap_vs_can = float(clap_model.compute_similarity(
                caption_emb, can_emb)[0][0].detach().cpu())

            results.append({
                'region_id':       row['region_id'],
                'canonical_class': cls,
                'raw_label':       row['raw_label'],
                'zs_top_class':    zs_class,
                'zs_sim':          round(zs_sim, 3),
                'zs_correct':      zs_correct,
                'caption':         caption,
                'cap_vs_raw':      round(cap_vs_raw, 3),
                'cap_vs_can':      round(cap_vs_can, 3),
                'cap_match_raw':   cap_vs_raw >= sim_threshold,
                'cap_match_can':   cap_vs_can >= sim_threshold,
            })

        except Exception as ex:
            print(f"  ERROR {row['region_id']}: {ex}")
        finally:
            os.unlink(tmp.name)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ann)} done...")

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path} ({len(df)} rows)")
    return df, audio_emb_store


def run_retrieval(clap_model, audio_emb_store, output_path):
    print("Step 4 — Text-to-audio retrieval")
    region_ids  = list(audio_emb_store.keys())
    stored_embs = [audio_emb_store[r]['emb'] for r in region_ids]
    stored_cls  = [audio_emb_store[r]['cls'] for r in region_ids]

    retrieval_results = []
    for r_id, data in audio_emb_store.items():
        query_emb = clap_model.get_text_embeddings([data['raw_label']])
        sims = []
        for emb in stored_embs:
            s = float(clap_model.compute_similarity(
                emb, query_emb)[0][0].detach().cpu())
            sims.append(s)
        sims = np.array(sims)
        self_idx = region_ids.index(r_id)
        sims[self_idx] = -999
        top_idx  = int(np.argmax(sims))
        top_cls  = stored_cls[top_idx]
        correct  = (top_cls == data['cls'])
        retrieval_results.append({
            'region_id':     r_id,
            'query_class':   data['cls'],
            'retrieved_cls': top_cls,
            'correct':       correct,
        })

    ret_df = pd.DataFrame(retrieval_results)
    ret_df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")
    return ret_df

def print_results(df, ret_df):
    print("="*65)
    print("FULL MICROSOFT CLAP RESULTS — September 2020 (first 30 min)")
    print("="*65)
    print(f"\n  {'Class':<15} {'N':>4}  {'ZS%':>6}  {'Caption%':>9}  {'Retrieval%':>11}")
    print(f"  {'-'*50}")

    for cls in sorted(df['canonical_class'].unique()):
        sub     = df[df['canonical_class'] == cls]
        ret     = ret_df[ret_df['query_class'] == cls]
        zs_pct  = sub['zs_correct'].mean() * 100
        cap_pct = sub['cap_match_raw'].mean() * 100
        ret_pct = ret['correct'].mean() * 100 if len(ret) else 0
        print(f"  {cls:<15} {len(sub):>4}  {zs_pct:>5.1f}%  {cap_pct:>8.1f}%  {ret_pct:>10.1f}%")


if __name__ == "__main__":
    check_device()
    
    clap_model, clapcap = load_models(use_cuda=torch.cuda.is_available())
    
    audio, sr, ann = load_data(
        AUDIO_PATH, ANNOTATIONS_PATH, RECORDING_DATE, MAX_DURATION_MIN
    )
    
    classes, descriptions, class_text_embs = embed_classes(
        clap_model, CLASS_DESCRIPTIONS
    )
    
    df, audio_emb_store = run_inference(
        audio, sr, ann, clap_model, clapcap, class_text_embs,
        classes, output_path="results/msclap_20200923.csv"
    )
    
    ret_df = run_retrieval(
        clap_model, audio_emb_store,
        output_path="results/msclap_retrieval_20200923.csv"
    )
    
    print_results(df, ret_df)