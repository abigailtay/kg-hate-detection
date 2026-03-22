"""
Explore dataset structure before preprocessing
"""
from datasets import load_from_disk
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/raw")

def explore_hatexplain():
    print("\n" + "="*60)
    print("HATEXPLAIN")
    print("="*60)
    ds = load_from_disk(DATA_DIR / "hatexplain")
    print(f"Splits: {list(ds.keys())}")
    print(f"Train size: {len(ds['train'])}")
    print(f"Columns: {ds['train'].column_names}")
    print(f"\nSample row:")
    sample = ds['train'][0]
    for k, v in sample.items():
        print(f"  {k}: {v}")
    
    # Labels are inside annotators['label'] - need majority vote
    # 0=hatespeech, 1=normal, 2=offensive
    all_labels = []
    for row in ds['train']:
        label_votes = row['annotators']['label']
        majority = max(set(label_votes), key=label_votes.count)
        all_labels.append(majority)
    print(f"\nLabel distribution (majority vote): {Counter(all_labels)}")
    print("  0=hatespeech, 1=normal, 2=offensive")

def explore_sbic():
    print("\n" + "="*60)
    print("SBIC (Social Bias Frames)")
    print("="*60)
    ds = load_from_disk(DATA_DIR / "sbic")
    print(f"Splits: {list(ds.keys())}")
    print(f"Train size: {len(ds['train'])}")
    print(f"Columns: {ds['train'].column_names}")
    print(f"\nSample row:")
    sample = ds['train'][0]
    for k, v in sample.items():
        print(f"  {k}: {v}")

def explore_implicit_hate():
    print("\n" + "="*60)
    print("IMPLICIT HATE")
    print("="*60)
    ds = load_from_disk(DATA_DIR / "implicit_hate")
    print(f"Splits: {list(ds.keys())}")
    print(f"Train size: {len(ds['train'])}")
    print(f"Columns: {ds['train'].column_names}")
    print(f"\nSample row:")
    sample = ds['train'][0]
    for k, v in sample.items():
        print(f"  {k}: {v}")
    
    # Check for class/label column
    for col in ds['train'].column_names:
        if 'class' in col.lower() or 'label' in col.lower() or 'implicit' in col.lower():
            vals = [row[col] for row in ds['train']]
            print(f"\n{col} distribution: {Counter(vals)}")

def explore_measuring_hate():
    print("\n" + "="*60)
    print("MEASURING HATE SPEECH")
    print("="*60)
    ds = load_from_disk(DATA_DIR / "measuring_hate_speech")
    print(f"Splits: {list(ds.keys())}")
    print(f"Train size: {len(ds['train'])}")
    print(f"Columns: {ds['train'].column_names}")
    print(f"\nSample row:")
    sample = ds['train'][0]
    for k, v in sample.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    explore_hatexplain()
    explore_sbic()
    explore_implicit_hate()
    explore_measuring_hate()
