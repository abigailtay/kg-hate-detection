"""
Preprocess crisis datasets and add to combined dataset
Maps to Label 1.3 (Crisis/Self-Harm)
"""
from datasets import load_from_disk, Dataset, DatasetDict, Features, Value
from pathlib import Path
from collections import Counter

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Define consistent schema
SCHEMA = Features({
    'text': Value('string'),
    'label': Value('int64'),
    'label_name': Value('string'),
    'source': Value('string'),
})


def process_depression_detection():
    print("\n[1/2] Processing depression-detection...")
    ds = load_from_disk(RAW_DIR / "depression_detection")
    
    examples = []
    for row in ds['train']:
        if row['label'] == 1:  # 1 = crisis
            examples.append({
                "text": str(row['text']),
                "label": 3,
                "label_name": "crisis",
                "source": "depression_detection",
            })
    train_ds = Dataset.from_list(examples, features=SCHEMA) if examples else None
    
    examples = []
    for row in ds['test']:
        if row['label'] == 1:
            examples.append({
                "text": str(row['text']),
                "label": 3,
                "label_name": "crisis",
                "source": "depression_detection",
            })
    test_ds = Dataset.from_list(examples, features=SCHEMA) if examples else None
    
    print(f"  ✓ Crisis examples - Train: {len(train_ds) if train_ds else 0}, Test: {len(test_ds) if test_ds else 0}")
    return train_ds, test_ds


def process_suicide_prediction():
    print("\n[2/2] Processing suicide_prediction...")
    ds = load_from_disk(RAW_DIR / "suicide_prediction")
    
    examples = []
    for row in ds['train']:
        if row['label'] == 'suicide':  # String label
            examples.append({
                "text": str(row['text']),
                "label": 3,
                "label_name": "crisis",
                "source": "suicide_prediction",
            })
    train_ds = Dataset.from_list(examples, features=SCHEMA) if examples else None
    
    examples = []
    for row in ds['test']:
        if row['label'] == 'suicide':
            examples.append({
                "text": str(row['text']),
                "label": 3,
                "label_name": "crisis",
                "source": "suicide_prediction",
            })
    test_ds = Dataset.from_list(examples, features=SCHEMA) if examples else None
    
    print(f"  ✓ Crisis examples - Train: {len(train_ds) if train_ds else 0}, Test: {len(test_ds) if test_ds else 0}")
    return train_ds, test_ds


def update_combined_dataset():
    print("\n" + "="*60)
    print("Updating combined dataset with crisis data...")
    print("="*60)
    
    dep_train, dep_test = process_depression_detection()
    sui_train, sui_test = process_suicide_prediction()
    
    # Combine crisis datasets
    crisis_train_list = []
    crisis_test_list = []
    
    if dep_train:
        crisis_train_list.extend(list(dep_train))
    if sui_train:
        crisis_train_list.extend(list(sui_train))
    if dep_test:
        crisis_test_list.extend(list(dep_test))
    if sui_test:
        crisis_test_list.extend(list(sui_test))
    
    crisis_train = Dataset.from_list(crisis_train_list, features=SCHEMA)
    crisis_test = Dataset.from_list(crisis_test_list, features=SCHEMA)
    
    print(f"\n  Total crisis - Train: {len(crisis_train)}, Test: {len(crisis_test)}")
    
    # Load existing combined dataset
    combined = load_from_disk(PROCESSED_DIR / "combined")
    train_labels = [x['label'] for x in combined['train']]
    print(f"\n  Current combined train distribution: {Counter(train_labels)}")
    
    # Downsample crisis
    TARGET_CRISIS = 25000
    TARGET_CRISIS_VAL = 3000
    TARGET_CRISIS_TEST = 3000
    
    crisis_train_sampled = crisis_train.shuffle(seed=42).select(range(min(TARGET_CRISIS, len(crisis_train))))
    crisis_test_sampled = crisis_test.shuffle(seed=42).select(range(min(TARGET_CRISIS_TEST, len(crisis_test))))
    crisis_val_sampled = crisis_test.shuffle(seed=43).select(range(min(TARGET_CRISIS_VAL, len(crisis_test))))
    
    print(f"\n  Downsampled crisis - Train: {len(crisis_train_sampled)}, Val: {len(crisis_val_sampled)}, Test: {len(crisis_test_sampled)}")
    
    # Convert existing combined to same schema
    def convert_combined(split_data):
        examples = []
        for row in split_data:
            examples.append({
                "text": str(row['text']),
                "label": int(row['label']),
                "label_name": str(row['label_name']),
                "source": str(row['source']),
            })
        return examples
    
    # Merge
    new_train_list = convert_combined(combined['train']) + list(crisis_train_sampled)
    new_val_list = convert_combined(combined['validation']) + list(crisis_val_sampled)
    new_test_list = convert_combined(combined['test']) + list(crisis_test_sampled)
    
    new_train = Dataset.from_list(new_train_list, features=SCHEMA).shuffle(seed=42)
    new_val = Dataset.from_list(new_val_list, features=SCHEMA).shuffle(seed=42)
    new_test = Dataset.from_list(new_test_list, features=SCHEMA).shuffle(seed=42)
    
    new_combined = DatasetDict({
        'train': new_train,
        'validation': new_val,
        'test': new_test
    })
    new_combined.save_to_disk(PROCESSED_DIR / "combined")
    
    print(f"\n✓ Updated combined dataset:")
    for split in new_combined:
        labels = [x['label'] for x in new_combined[split]]
        print(f"  {split}: {len(new_combined[split])} examples")
        print(f"    {Counter(labels)}")
    
    return new_combined


if __name__ == "__main__":
    print("="*60)
    print("CRISIS DATA PREPROCESSING")
    print("="*60)
    update_combined_dataset()
    print("\n" + "="*60)
    print("✓ Crisis data added to combined dataset!")
    print("="*60)
