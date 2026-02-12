import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class VulnerabilityDataset(Dataset):
    def __init__(self, codes,labels,tokenizer,max_length = 768 ):
        self.codes = codes
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.codes)

    def __getitem__(self,idx):

        code = self.codes[idx]
        label = self.labels[idx]
        tokenized, attention_mask = self.tokenizer.encode(code, self.max_length)

        return {
            'tokenized': torch.tensor(tokenized, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

def prepare_paired_dataloaders(df, tokenizer, batch_size=16, max_length=768, test_size=0.3):
    print("\n" + "="*80)
    print("PREPARING PAIRED DATALOADERS")
    print("="*80)
    
    # ==============================================================
    # STEP 1: Understand the dataset structure
    # ==============================================================
    
    total_samples = len(df)
    vuln_count = len(df[df['label'] == 1])
    clean_count = len(df[df['label'] == 0])
    
    print(f"\nDataset info:")
    print(f"  Total samples: {total_samples}")
    print(f"  Vulnerable (label=1): {vuln_count}")
    print(f"  Clean (label=0): {clean_count}")
    
    # Number of pairs
    num_pairs = min(vuln_count, clean_count)
    print(f"  Number of pairs: {num_pairs}")
    
    # ==============================================================
    # STEP 2: Create pair indices
    # ==============================================================
    
    # Pair indices: [0, 1, 2, ..., num_pairs-1]
    # Each number represents one vulnerable-patched pair
    pair_indices = np.arange(num_pairs)
    
    print(f"\nPair indices: 0 to {num_pairs-1}")
    
    # ==============================================================
    # STEP 3: Split pairs (not individual samples!)
    # ==============================================================
    
    # Split 1: 70% train, 30% temp
    train_pair_idx, temp_pair_idx = train_test_split(
        pair_indices,
        test_size=test_size,  # 0.3 = 30%
        random_state=42        # For reproducibility
    )
    
    # Split 2: Split temp into 50/50 → val and test
    # 30% → 15% val, 15% test
    val_pair_idx, test_pair_idx = train_test_split(
        temp_pair_idx,
        test_size=0.5,
        random_state=42
    )
    
    print(f"\nPair split:")
    print(f"  Train pairs: {len(train_pair_idx)} ({len(train_pair_idx)/num_pairs*100:.1f}%)")
    print(f"  Val pairs:   {len(val_pair_idx)} ({len(val_pair_idx)/num_pairs*100:.1f}%)")
    print(f"  Test pairs:  {len(test_pair_idx)} ({len(test_pair_idx)/num_pairs*100:.1f}%)")
    
    # ==============================================================
    # STEP 4: Convert pair indices to row indices
    # ==============================================================
    
    def pairs_to_row_indices(pair_idx_array):
        # Get vulnerable row indices (first half)
        vuln_indices = pair_idx_array
        
        # Get patched row indices (second half)
        patch_indices = pair_idx_array + num_pairs
        
        # Combine both
        all_indices = np.concatenate([vuln_indices, patch_indices])
        
        # Shuffle so vulnerable and patched are mixed
        np.random.seed(42)
        np.random.shuffle(all_indices)
        
        return all_indices
    
    # Convert pair indices to row indices
    train_indices = pairs_to_row_indices(train_pair_idx)
    val_indices = pairs_to_row_indices(val_pair_idx)
    test_indices = pairs_to_row_indices(test_pair_idx)
    
    # ==============================================================
    # STEP 5: Create DataFrames for each split
    # ==============================================================
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    print(f"\nSample split:")
    print(f"  Train: {len(train_df)} samples " + 
          f"({len(train_df[train_df['label']==1])} vuln, {len(train_df[train_df['label']==0])} clean)")
    print(f"  Val:   {len(val_df)} samples " + 
          f"({len(val_df[val_df['label']==1])} vuln, {len(val_df[val_df['label']==0])} clean)")
    print(f"  Test:  {len(test_df)} samples " + 
          f"({len(test_df[test_df['label']==1])} vuln, {len(test_df[test_df['label']==0])} clean)")
    
    # ==============================================================
    # STEP 6: Verify no data leakage
    # ==============================================================
    
    # Extract function names (remove _patched suffix to compare)
    train_funcs = set(train_df['function_name'].str.replace('_patched', '', regex=False))
    val_funcs = set(val_df['function_name'].str.replace('_patched', '', regex=False))
    test_funcs = set(test_df['function_name'].str.replace('_patched', '', regex=False))
    
    # Check for overlaps
    train_val_overlap = train_funcs & val_funcs
    train_test_overlap = train_funcs & test_funcs
    val_test_overlap = val_funcs & test_funcs
    
    print(f"\n✓ Data leakage check:")
    print(f"  Train-Val overlap:  {len(train_val_overlap)} functions")
    print(f"  Train-Test overlap: {len(train_test_overlap)} functions")
    print(f"  Val-Test overlap:   {len(val_test_overlap)} functions")
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("No leakage detected - pairs properly separated!")
    else:
        print("WARNING: Data leakage detected!")
    
    # ==============================================================
    # STEP 7: Create Dataset objects
    # ==============================================================
    
    print(f"\nCreating datasets with max_length={max_length}...")
    
    train_dataset = VulnerabilityDataset(
        codes=train_df['code'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = VulnerabilityDataset(
        codes=val_df['code'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = VulnerabilityDataset(
        codes=test_df['code'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # ==============================================================
    # STEP 8: Create DataLoader objects
    # ==============================================================
    
    print(f"Creating dataloaders with batch_size={batch_size}...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # Shuffle for training
        num_workers=0,  # 0 for Windows, can increase on Linux
        pin_memory=torch.cuda.is_available()  # Speed up GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for validation
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for testing
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\n✅ Dataloaders ready!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader