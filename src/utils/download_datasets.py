"""
Dataset downloader for KG-Augmented Hate Speech Detection
Run on CPU - no GPU needed
"""
import os
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_hatexplain():
    """HateXplain: Primary Dim 1 training data"""
    print("\n[1/4] Downloading HateXplain...")
    ds = load_dataset("hatexplain", trust_remote_code=True)
    ds.save_to_disk(DATA_DIR / "hatexplain")
    print(f"  ✓ Saved to {DATA_DIR}/hatexplain")
    print(f"  Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

def download_sbic():
    """Social Bias Inference Corpus: Implicit bias"""
    print("\n[2/4] Downloading SBIC...")
    ds = load_dataset("social_bias_frames", trust_remote_code=True)
    ds.save_to_disk(DATA_DIR / "sbic")
    print(f"  ✓ Saved to {DATA_DIR}/sbic")

def download_implicit_hate():
    """Implicit Hate Corpus"""
    print("\n[3/4] Downloading Implicit Hate Corpus...")
    ds = load_dataset("SALT-NLP/ImplicitHate", trust_remote_code=True)
    ds.save_to_disk(DATA_DIR / "implicit_hate")
    print(f"  ✓ Saved to {DATA_DIR}/implicit_hate")

def download_measuring_hate():
    """Measuring Hate Speech dataset"""
    print("\n[4/4] Downloading Measuring Hate Speech...")
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", trust_remote_code=True)
    ds.save_to_disk(DATA_DIR / "measuring_hate_speech")
    print(f"  ✓ Saved to {DATA_DIR}/measuring_hate_speech")

if __name__ == "__main__":
    print("=" * 50)
    print("DATASET DOWNLOAD - Dim 1 Core Datasets")
    print("=" * 50)
    
    download_hatexplain()
    download_sbic()
    download_implicit_hate()
    download_measuring_hate()
    
    print("\n" + "=" * 50)
    print("✓ All datasets downloaded to data/raw/")
    print("=" * 50)
