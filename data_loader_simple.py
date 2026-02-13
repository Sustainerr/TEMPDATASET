import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class VulnerabilityDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length=768):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]
        
        tokenized, attention_mask = self.tokenizer.encode(code, self.max_length)
        
        return {
            'tokenized': torch.tensor(tokenized, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_simple_dataloaders(df, tokenizer, batch_size=16, max_length=640):
    """
    Simple random split without pair awareness
    No data leakage - completely separate functions in train/val/test
    """
    
    print("\n" + "="*80)
    print("PREPARING DATALOADERS (Simple Random Split)")
    print("="*80)
    
    # 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['label']
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df[train_df['label']==1])} vuln, {len(train_df[train_df['label']==0])} clean)")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df[val_df['label']==1])} vuln, {len(val_df[val_df['label']==0])} clean)")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df[test_df['label']==1])} vuln, {len(test_df[test_df['label']==0])} clean)")
    
    # Create datasets
    train_dataset = VulnerabilityDataset(
        train_df['code'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length
    )
    
    val_dataset = VulnerabilityDataset(
        val_df['code'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length
    )
    
    test_dataset = VulnerabilityDataset(
        test_df['code'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"\n✓ Dataloaders ready!")
    print(f"  Batch size: {batch_size}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
