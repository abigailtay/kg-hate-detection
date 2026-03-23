"""
Download public crisis/self-harm datasets (no IRB required)
These map to Label 1.3 in our taxonomy
"""
from datasets import load_dataset, load_from_disk
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_depression_detection():
    """Reddit depression/suicide posts - 200K examples"""
    print("\n[1/2] Downloading depression-detection...")
    ds = load_dataset("thePixel42/depression-detection")
    ds.save_to_disk(DATA_DIR / "depression_detection")
    print(f"  ✓ Saved to {DATA_DIR}/depression_detection")
    print(f"  Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    # Label 0 = non-crisis, Label 1 = crisis (from r/SuicideWatch, r/depression)
    from collections import Counter
    print(f"  Train labels: {Counter(ds['train']['label'])}")

def download_suicide_prediction():
    """Reddit suicide prediction - 232K examples"""
    print("\n[2/2] Downloading suicide_prediction_dataset_phr...")
    ds = load_dataset("vibhorag101/suicide_prediction_dataset_phr")
    ds.save_to_disk(DATA_DIR / "suicide_prediction")
    print(f"  ✓ Saved to {DATA_DIR}/suicide_prediction")
    print(f"  Train: {len(ds['train'])}, Test: {len(ds['test'])}")

if __name__ == "__main__":
    print("=" * 60)
    print("CRISIS DATASET DOWNLOAD (Label 1.3)")
    print("=" * 60)
    
    download_depression_detection()
    download_suicide_prediction()
    
    print("\n" + "=" * 60)
    print("✓ Crisis datasets downloaded!")
    print("=" * 60)
