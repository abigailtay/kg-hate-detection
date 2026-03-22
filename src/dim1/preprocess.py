"""
Preprocess datasets and map to Dim 1 labels:
  1.0 = Non-Harmful
  1.1 = Identity-Based Hate
  1.2 = Interpersonal Abuse
  1.3 = Crisis/Self-Harm (synthetic only - not in these datasets)
"""
from datasets import load_from_disk, Dataset, DatasetDict
from pathlib import Path
from collections import Counter
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Label mapping
LABELS = {
    "1.0": 0,  # Non-Harmful
    "1.1": 1,  # Identity-Based Hate
    "1.2": 2,  # Interpersonal Abuse
    "1.3": 3,  # Crisis/Self-Harm
}
LABEL_NAMES = ["non_harmful", "identity_hate", "interpersonal_abuse", "crisis"]


def safe_float(val, default=0.0):
    """Safely convert to float"""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def process_hatexplain():
    """
    HateXplain: majority vote from annotators
    0=hatespeech→1.1, 1=normal→1.0, 2=offensive→1.2
    """
    print("\n[1/4] Processing HateXplain...")
    ds = load_from_disk(RAW_DIR / "hatexplain")
    
    def convert(example):
        # Get majority vote
        votes = example['annotators']['label']
        majority = max(set(votes), key=votes.count)
        
        # Map: 0=hatespeech→1.1, 1=normal→1.0, 2=offensive→1.2
        label_map = {0: 1, 1: 0, 2: 2}  # to our indices
        
        # Join tokens to text
        text = " ".join(example['post_tokens'])
        
        # Get target groups if identity hate
        targets = []
        if majority == 0:  # hatespeech
            for annotator_targets in example['annotators']['target']:
                targets.extend(annotator_targets)
            targets = list(set(targets))
        
        return {
            "text": text,
            "label": label_map[majority],
            "label_name": LABEL_NAMES[label_map[majority]],
            "targets": targets,
            "source": "hatexplain",
            "rationales": example['rationales'],  # keep for XAI
        }
    
    processed = {}
    for split in ds.keys():
        processed[split] = ds[split].map(convert, remove_columns=ds[split].column_names)
    
    result = DatasetDict(processed)
    result.save_to_disk(PROCESSED_DIR / "hatexplain")
    
    # Stats
    train_labels = [x['label'] for x in result['train']]
    print(f"  ✓ Saved to {PROCESSED_DIR}/hatexplain")
    print(f"  Train distribution: {Counter(train_labels)}")
    print(f"  Sample: {result['train'][0]['text'][:80]}...")
    return result


def process_sbic():
    """
    SBIC: Use offensiveYN and targetMinority
    offensive + has target → 1.1, offensive + no target → 1.2, not offensive → 1.0
    """
    print("\n[2/4] Processing SBIC...")
    ds = load_from_disk(RAW_DIR / "sbic")
    
    def convert(example):
        offensive = safe_float(example['offensiveYN'], 0.0)
        target = str(example['targetMinority'] or "").strip()
        text = str(example['post'] or "").strip()
        
        if not text:  # skip empty posts
            return None
            
        if offensive > 0.5:
            if target:  # has identity target
                label = 1  # 1.1 identity hate
            else:
                label = 2  # 1.2 interpersonal abuse
        else:
            label = 0  # 1.0 non-harmful
        
        return {
            "text": text,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "targets": [target] if target else [],
            "source": "sbic",
        }
    
    processed = {}
    for split in ds.keys():
        mapped = ds[split].map(convert, remove_columns=ds[split].column_names)
        # Filter out None results
        processed[split] = mapped.filter(lambda x: x['text'] is not None and len(x['text']) > 0)
    
    result = DatasetDict(processed)
    result.save_to_disk(PROCESSED_DIR / "sbic")
    
    train_labels = [x['label'] for x in result['train']]
    print(f"  ✓ Saved to {PROCESSED_DIR}/sbic")
    print(f"  Train distribution: {Counter(train_labels)}")
    return result


def process_implicit_hate():
    """
    Implicit Hate: All examples are implicit hate → 1.1
    """
    print("\n[3/4] Processing Implicit Hate...")
    ds = load_from_disk(RAW_DIR / "implicit_hate")
    
    def convert(example):
        return {
            "text": example['post'],
            "label": 1,  # All are 1.1 identity hate (implicit)
            "label_name": LABEL_NAMES[1],
            "targets": [],
            "source": "implicit_hate",
            "implicit_class": example['implicit_class'],
        }
    
    # Only has train split
    processed = ds['train'].map(convert, remove_columns=ds['train'].column_names)
    
    # Create train/val/test splits (80/10/10)
    splits = processed.train_test_split(test_size=0.2, seed=42)
    val_test = splits['test'].train_test_split(test_size=0.5, seed=42)
    
    result = DatasetDict({
        'train': splits['train'],
        'validation': val_test['train'],
        'test': val_test['test'],
    })
    result.save_to_disk(PROCESSED_DIR / "implicit_hate")
    
    print(f"  ✓ Saved to {PROCESSED_DIR}/implicit_hate")
    print(f"  Train: {len(result['train'])}, Val: {len(result['validation'])}, Test: {len(result['test'])}")
    return result


def process_measuring_hate():
    """
    Measuring Hate Speech: Use hate_speech_score and target_* columns
    score > 0 + has target → 1.1, score > 0 + no target → 1.2, score ≤ 0 → 1.0
    """
    print("\n[4/4] Processing Measuring Hate Speech...")
    ds = load_from_disk(RAW_DIR / "measuring_hate_speech")
    
    target_cols = [
        'target_race', 'target_religion', 'target_origin', 
        'target_gender', 'target_sexuality', 'target_age', 
        'target_disability', 'target_politics'
    ]
    
    def convert(example):
        score = safe_float(example['hate_speech_score'], 0.0)
        has_target = any(example.get(col, False) for col in target_cols)
        text = str(example['text'] or "").strip()
        
        if not text:
            return None
        
        if score > 0:
            if has_target:
                label = 1  # 1.1 identity hate
            else:
                label = 2  # 1.2 interpersonal abuse
        else:
            label = 0  # 1.0 non-harmful
        
        # Collect which targets
        targets = [col.replace('target_', '') for col in target_cols if example.get(col, False)]
        
        return {
            "text": text,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "targets": targets,
            "source": "measuring_hate",
            "hate_speech_score": score,
        }
    
    # Only has train split - need to dedupe by comment_id first
    df = ds['train'].to_pandas()
    df = df.drop_duplicates(subset=['comment_id'])
    deduped = Dataset.from_pandas(df)
    
    mapped = deduped.map(convert, remove_columns=deduped.column_names)
    processed = mapped.filter(lambda x: x['text'] is not None and len(x['text']) > 0)
    
    # Create train/val/test splits (80/10/10)
    splits = processed.train_test_split(test_size=0.2, seed=42)
    val_test = splits['test'].train_test_split(test_size=0.5, seed=42)
    
    result = DatasetDict({
        'train': splits['train'],
        'validation': val_test['train'],
        'test': val_test['test'],
    })
    result.save_to_disk(PROCESSED_DIR / "measuring_hate")
    
    train_labels = [x['label'] for x in result['train']]
    print(f"  ✓ Saved to {PROCESSED_DIR}/measuring_hate")
    print(f"  Train distribution: {Counter(train_labels)}")
    return result


def create_combined_dataset():
    """
    Combine all processed datasets into one unified dataset
    """
    print("\n" + "="*60)
    print("Creating combined dataset...")
    print("="*60)
    
    datasets = []
    for name in ["hatexplain", "sbic", "implicit_hate", "measuring_hate"]:
        path = PROCESSED_DIR / name
        if path.exists():
            ds = load_from_disk(path)
            datasets.append((name, ds))
    
    # Combine train splits
    combined = {'train': [], 'validation': [], 'test': []}
    
    for name, ds in datasets:
        for split in ['train', 'validation', 'test']:
            if split in ds:
                for example in ds[split]:
                    combined[split].append({
                        'text': example['text'],
                        'label': example['label'],
                        'label_name': example['label_name'],
                        'source': example['source'],
                    })
    
    # Convert to Dataset
    result = DatasetDict({
        split: Dataset.from_list(examples) 
        for split, examples in combined.items()
    })
    
    result.save_to_disk(PROCESSED_DIR / "combined")
    
    print(f"\n✓ Combined dataset saved to {PROCESSED_DIR}/combined")
    for split in result:
        labels = [x['label'] for x in result[split]]
        print(f"  {split}: {len(result[split])} examples")
        print(f"    Distribution: {Counter(labels)}")
    
    return result


if __name__ == "__main__":
    print("="*60)
    print("PREPROCESSING - Map to Dim 1 Labels")
    print("="*60)
    
    process_hatexplain()
    process_sbic()
    process_implicit_hate()
    process_measuring_hate()
    create_combined_dataset()
    
    print("\n" + "="*60)
    print("✓ All datasets processed!")
    print("="*60)
