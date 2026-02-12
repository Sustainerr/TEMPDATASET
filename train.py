import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
from datetime import datetime

# Import your modules
from model import SafeCodeModel
from model import TreesitterTokenizer
from dataloader import prepare_paired_dataloaders


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['tokenized'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss = total_loss + loss.item()

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
        
    return avg_loss, accuracy
    
def evaluate(model, dataloader, criterion, device):
    
    # Set model to evaluation mode
    # This disables dropout and uses batch norm in inference mode
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation (saves memory and speeds up)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            input_ids = batch['tokenized'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass only (no backward pass)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

def main():

    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    # Paths
    TSV_PATH = 'megavul_dataset.tsv'
    OUTPUT_DIR = 'outputs'
    
    # Model hyperparameters - BALANCED CONFIG
    MAX_LENGTH = 512        # More context
    BATCH_SIZE = 16         
    EMBED_DIM = 256         # More capacity
    HIDDEN_SIZE = 256       # More capacity
    NUM_LAYERS = 2          # Deeper model
    NHEAD = 8               
    DROPOUT = 0.3           

    # Training hyperparameters
    LEARNING_RATE = 1e-4    
    NUM_EPOCHS = 30         
    PATIENCE = 7            
    MIN_VOCAB_FREQ = 2

    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print("="*80)
    print("VULNERABILITY DETECTION MODEL TRAINING")
    print("="*80)
    print(f"Output directory: {run_dir}\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ========================================
    # STEP 1: LOAD DATA
    # ========================================
    
    print(f"\n{'='*80}")
    print("STEP 1: Loading Dataset")
    print("="*80)
    
    df = pd.read_csv(TSV_PATH, sep='\t')
    print(f"✓ Dataset loaded: {len(df)} samples")
    print(f"  Vulnerable: {len(df[df['label']==1])}")
    print(f"  Clean: {len(df[df['label']==0])}")
    
    # ========================================
    # STEP 2: BUILD TOKENIZER
    # ========================================
    
    print(f"\n{'='*80}")
    print("STEP 2: Building Tokenizer")
    print("="*80)
    
    tokenizer = TreesitterTokenizer()
    tokenizer.vocab(df['code'].tolist(), min_freq=MIN_VOCAB_FREQ)
    tokenizer.save(os.path.join(run_dir, 'tokenizer.pkl'))
    
    # ========================================
    # STEP 3: PREPARE DATALOADERS
    # ========================================
    
    print(f"\n{'='*80}")
    print("STEP 3: Preparing Dataloaders")
    print("="*80)
    
    train_loader, val_loader, test_loader = prepare_paired_dataloaders(
        df=df,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    
    # ========================================
    # STEP 4: INITIALIZE MODEL
    # ========================================
    
    print(f"\n{'='*80}")
    print("STEP 4: Initializing Model")
    print("="*80)
    
    model = SafeCodeModel(
        vocabSize=tokenizer.vocab_size,
        embeddedDim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        hiddenSize=HIDDEN_SIZE,
        nhead=NHEAD
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # ========================================
    # STEP 5: SETUP TRAINING
    # ========================================
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # ========================================
    # STEP 6: TRAINING LOOP
    # ========================================
    
    print(f"\n{'='*80}")
    print("STEP 5: Training")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1
        })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, os.path.join(run_dir, 'best_model.pth'))
            print("✓ Best model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(run_dir, 'training_history.csv'), index=False)
    
    # ========================================
    # STEP 7: FINAL EVALUATION
    # ========================================
    
    print(f"\n{'='*80}")
    print("STEP 6: Final Evaluation on Test Set")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(run_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Clean  Vulnerable")
    print(f"Actual Clean    {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Vuln     {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {run_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
