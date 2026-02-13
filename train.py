import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import SafeCodeModel, TreesitterTokenizer
from data_loader_simple import prepare_simple_dataloaders


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["tokenized"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["tokenized"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


# ========= THRESHOLD-BASED EVAL HELPERS =========


def eval_with_threshold(model, dataloader, device, threshold=0.5):
    """
    Evaluate using a probability threshold instead of argmax.
    Returns: loss, accuracy, precision, recall, f1, preds, labels
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Evaluating (th={threshold:.2f})"
        ):
            input_ids = batch["tokenized"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]  # P(vulnerable)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= threshold).astype(int)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary"
    )

    return avg_loss, acc, precision, recall, f1, preds, all_labels


def find_best_threshold(model, val_loader, device):
    """
    Try several thresholds on the validation set and pick the best F1.
    """
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP ON VALIDATION SET")
    print("=" * 80)
    print(f"{'th':<6} {'prec':<8} {'rec':<8} {'f1':<8} {'acc':<8}")
    print("-" * 40)

    best_f1 = -1.0
    best_th = 0.5

    for th in thresholds:
        _, acc, prec, rec, f1, _, _ = eval_with_threshold(
            model, val_loader, device, threshold=th
        )
        print(f"{th:<6.2f} {prec:<8.3f} {rec:<8.3f} {f1:<8.3f} {acc:<8.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print("\nBest threshold on val:", best_th, "F1 =", best_f1)
    return best_th


# ========= MAIN =========


def main():
    # ========================================
    # CONFIGURATION
    # ========================================

    TSV_PATH = "megavul_embedded_c_dataset.tsv"
    OUTPUT_DIR = "outputs"

    MAX_LENGTH = 256
    BATCH_SIZE = 64
    EMBED_DIM = 256
    HIDDEN_SIZE = 256
    NUM_LAYERS = 1
    NHEAD = 4
    DROPOUT = 0.3

    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 30
    PATIENCE = 5
    MIN_VOCAB_FREQ = 3

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print("VULNERABILITY DETECTION MODEL TRAINING")
    print("=" * 80)
    print(f"Output directory: {run_dir}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========================================
    # STEP 1: LOAD DATA
    # ========================================

    print(f"\n{'='*80}")
    print("STEP 1: Loading Dataset")
    print("=" * 80)

    df = pd.read_csv(TSV_PATH, sep="\t")
    print(f"✓ Dataset loaded: {len(df)} samples")
    print(f"  Vulnerable: {len(df[df['label'] == 1])}")
    print(f"  Clean:      {len(df[df['label'] == 0])}")

    # ========================================
    # STEP 2: BUILD TOKENIZER
    # ========================================

    print(f"\n{'='*80}")
    print("STEP 2: Building Tokenizer")
    print("=" * 80)

    tokenizer = TreesitterTokenizer()
    tokenizer.vocab(df["code"].tolist(), min_freq=MIN_VOCAB_FREQ)
    tokenizer.save(os.path.join(run_dir, "tokenizer.pkl"))

    # ========================================
    # STEP 3: PREPARE DATALOADERS
    # ========================================

    print(f"\n{'='*80}")
    print("STEP 3: Preparing Dataloaders")
    print("=" * 80)

    train_loader, val_loader, test_loader = prepare_simple_dataloaders(
        df=df,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )


        # ============================================
    # DEBUG: INSPECT DATA
    # ============================================
    print("\n" + "="*80)
    print("DEBUG: INSPECTING BATCH")
    print("="*80)

    batch = next(iter(train_loader))
    print("Tokenized shape:", batch["tokenized"].shape)
    print("Labels shape:", batch["labels"].shape)
    print("\nFirst 20 labels:", batch["labels"][:20])
    print("Label counts in batch:", batch["labels"].bincount())

    print("\nFirst sequence (first 50 tokens):")
    print(batch["tokenized"][0][:50])

    print("\nUnique tokens in first sequence:", len(batch["tokenized"][0].unique()))
    print("Some unique tokens:", batch["tokenized"][0].unique()[:20])

    print("\nTokenizer vocab size:", tokenizer.vocab_size)

    # SAFE VOCAB DEBUG – no .stoi assumption
    for attr in ["vocab", "itos", "id_to_token", "token_to_id", "token_to_idx"]:
        if hasattr(tokenizer, attr):
            val = getattr(tokenizer, attr)
            print(f"\nDEBUG: tokenizer.{attr} type:", type(val))
            if isinstance(val, dict):
                items = list(val.items())[:30]
            elif isinstance(val, (list, tuple)):
                items = list(enumerate(val[:30]))
            else:
                items = str(val)[:200]
            print(f"Sample from tokenizer.{attr}:")
            print(items)
            break
    else:
        print("\nDEBUG: tokenizer has no known vocab mapping attributes.")

    print("="*80)
    print("DEBUG COMPLETE - Review the output above")
    print("="*80)
    input("Press ENTER to continue training...")



    # ========================================
    # STEP 4: INITIALIZE MODEL
    # ========================================

    print(f"\n{'='*80}")
    print("STEP 4: Initializing Model")
    print("=" * 80)

    model = SafeCodeModel(
        vocabSize=tokenizer.vocab_size,
        embeddedDim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        hiddenSize=HIDDEN_SIZE,
        nhead=NHEAD,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # ========================================
    # STEP 5: SETUP TRAINING
    # ========================================

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ========================================
    # STEP 6: TRAINING LOOP
    # ========================================

    print(f"\n{'='*80}")
    print("STEP 5: Training")
    print("=" * 80)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*80}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(
            f"Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
            }
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                },
                os.path.join(run_dir, "best_model.pth"),
            )
            print("✓ Best model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(run_dir, "training_history.csv"), index=False
    )

    # ========================================
    # STEP 7: FINAL EVALUATION + THRESHOLD TUNING
    # ========================================

    print(f"\n{'='*80}")
    print("STEP 6: Final Evaluation on Test Set")
    print("=" * 80)

    checkpoint = torch.load(os.path.join(run_dir, "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # 1) Find best threshold on validation set
    best_th = find_best_threshold(model, val_loader, device)

    # 2) Evaluate on test set with default threshold 0.5
    print(f"\n{'-'*80}")
    print("Test with default threshold = 0.50")
    print("-" * 80)
    test_loss, test_acc, test_prec, test_rec, test_f1, preds_05, labels_05 = (
        eval_with_threshold(model, test_loader, device, threshold=0.5)
    )
    print(f"\nTest Results (th=0.50):")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")

    cm = confusion_matrix(labels_05, preds_05)
    print(f"\nConfusion Matrix (th=0.50):")
    print(f"                Predicted")
    print(f"              Clean  Vulnerable")
    print(f"Actual Clean    {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Vuln     {cm[1][0]:5d}  {cm[1][1]:5d}")

    # 3) Evaluate on test set with best threshold
    print(f"\n{'-'*80}")
    print(f"Test with optimized threshold = {best_th:.2f}")
    print("-" * 80)
    _, test_acc2, test_prec2, test_rec2, test_f12, preds_best, labels_best = (
        eval_with_threshold(model, test_loader, device, threshold=best_th)
    )
    print(f"\nTest Results (th={best_th:.2f}):")
    print(f"  Accuracy:  {test_acc2:.4f}")
    print(f"  Precision: {test_prec2:.4f}")
    print(f"  Recall:    {test_rec2:.4f}")
    print(f"  F1 Score:  {test_f12:.4f}")

    cm2 = confusion_matrix(labels_best, preds_best)
    print(f"\nConfusion Matrix (th={best_th:.2f}):")
    print(f"                Predicted")
    print(f"              Clean  Vulnerable")
    print(f"Actual Clean    {cm2[0][0]:5d}  {cm2[0][1]:5d}")
    print(f"       Vuln     {cm2[1][0]:5d}  {cm2[1][1]:5d}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
