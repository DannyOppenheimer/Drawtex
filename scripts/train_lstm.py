import json
import sys
from pathlib import Path

# Project root directory (Drawtex/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import random
import joblib

from models.lstm import StrokeClassifierLSTM
from models.feature_extraction import (
    INPUT_DIM,
    extract_features,
    pad_sequence,
    smooth_predictions,
)

LABEL_MAP = {"text": 0, "math": 1, "diagram": 2}
HIDDEN_DIM = 64
OUTPUT_DIM = 3
BATCH_SIZE = 32

DATA_PATH = str(PROJECT_ROOT / "data" / "collected_strokes.json")
WEIGHTS_DIR = PROJECT_ROOT / "weights"
SCALER_PATH = str(WEIGHTS_DIR / "scaler.save")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(67)


def evaluate(model, loader, device):
    """
    Evaluates the model with Majority Vote Smoothing.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, batch_lengths in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            logits = model(batch_x, batch_lengths)
            preds = torch.argmax(logits, dim=2)  # [Batch Size, Max Seq Len]

            # We must process each sequence individually to smooth correctly
            # and ignore padding.
            for i in range(len(batch_y)):
                length = batch_lengths[i].item()  # Actual length of this sequence

                # Extract the valid part of the sequence (ignore padding)
                raw_pred_seq = preds[i, :length].cpu().numpy()
                target_seq = batch_y[i, :length].cpu().numpy()

                # Apply Smoothing (Majority Vote)
                smoothed_pred_seq = smooth_predictions(raw_pred_seq, window_size=5)

                all_preds.extend(smoothed_pred_seq)
                all_targets.extend(target_seq)

    return f1_score(all_targets, all_preds, average="weighted"), all_preds, all_targets


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure weights directory exists
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(DATA_PATH, "r") as f:
            raw_sessions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        exit()

    X_list = []
    Y_list = []

    for session in raw_sessions:
        feats, labs = extract_features(session)
        if len(feats) > 0:
            X_list.append(feats)
            Y_list.append(labs)

    if not X_list:
        print("No valid stroke data found.")
        exit()

    all_features = np.vstack(X_list)
    scaler = StandardScaler()
    scaler.fit(all_features)
    X_list_normalized = [scaler.transform(x) for x in X_list]

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to '{SCALER_PATH}'")

    data = list(zip(X_list_normalized, Y_list))
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, collate_fn=pad_sequence, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, collate_fn=pad_sequence, shuffle=False
    )

    all_labels = [label for session in Y_list for label in session]
    class_counts = np.bincount(all_labels)
    total_samples = sum(class_counts)

    # Compute weights: Inverse frequency
    class_weights = torch.tensor(
        [total_samples / c for c in class_counts], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum()  # Normalize roughly
    class_weights = class_weights.to(device)

    num_epochs = 125
    best_f1 = 0.0
    best_epoch = 0
    best_train_loss = 0.0

    model = StrokeClassifierLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    print(
        f"Starting training on {len(train_data)} sessions, validating on {len(val_data)} sessions."
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_x, batch_y, batch_lengths in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, batch_lengths)

            loss = criterion(logits.view(-1, OUTPUT_DIM), batch_y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # The evaluate function now returns the F1 of the SMOOTHED predictions
        val_f1, _, _ = evaluate(model, val_loader, device)
        avg_loss = train_loss / len(train_loader)
        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val F1 (Smoothed): {val_f1:.4f} | LR: {current_lr:.6f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_train_loss = avg_loss
            best_state = model.state_dict()
            print(f"  -> New best (F1: {best_f1:.4f})")

    f1_digits = f"{best_f1:.4f}".split(".")[1]
    model_save_path = str(WEIGHTS_DIR / f"0_{f1_digits}_best_model.pth")
    torch.save(best_state, model_save_path)

    # ── Report Card ──────────────────────────────────────────────────────
    REVERSE_LABEL_MAP = {0: "text", 1: "math", 2: "diagram"}
    final_lr = optimizer.param_groups[0]["lr"]
    final_train_loss = avg_loss  # from last epoch

    # Load best model for evaluation
    model.load_state_dict(best_state)

    val_f1, val_preds, val_targets = evaluate(model, val_loader, device)
    train_f1, _, _ = evaluate(model, train_loader, device)

    # Per-class stroke counts
    train_labels = [l for _, y in train_data for l in y]
    val_labels = [l for _, y in val_data for l in y]

    print("\n" + "=" * 60)
    print("  TRAINING REPORT CARD")
    print("=" * 60)

    # Dataset overview
    print(f"\n{'─' * 40}")
    print("  DATASET OVERVIEW")
    print(f"{'─' * 40}")
    print(f"  Total sessions:    {len(X_list)}")
    print(f"  Train sessions:    {len(train_data)}")
    print(f"  Val sessions:      {len(val_data)}")
    print(f"  Total strokes:     {total_samples}")
    for cls_id in range(OUTPUT_DIM):
        name = REVERSE_LABEL_MAP[cls_id]
        count = class_counts[cls_id] if cls_id < len(class_counts) else 0
        pct = 100 * count / total_samples
        print(f"    {name:>8}: {count:>6} ({pct:.1f}%)")

    # Training dynamics
    print(f"\n{'─' * 40}")
    print("  TRAINING DYNAMICS")
    print(f"{'─' * 40}")
    print(f"  Best epoch:        {best_epoch} / {num_epochs}")
    print(f"  Best val F1:       {best_f1:.4f}")
    print(f"  Train loss @ best: {best_train_loss:.4f}")
    print(f"  Train loss @ end:  {final_train_loss:.4f}")
    print(f"  Final LR:          {final_lr:.6f}")

    # Overfitting diagnostic
    print(f"\n{'─' * 40}")
    print("  OVERFITTING DIAGNOSTIC")
    print(f"{'─' * 40}")
    print(f"  Train F1 (best model): {train_f1:.4f}")
    print(f"  Val F1 (best model):   {val_f1:.4f}")
    print(f"  Gap:                   {train_f1 - val_f1:.4f}")

    # Per-class metrics
    print(f"\n{'─' * 40}")
    print("  PER-CLASS METRICS (Validation)")
    print(f"{'─' * 40}")
    target_names = [REVERSE_LABEL_MAP[i] for i in range(OUTPUT_DIM)]
    present_labels = sorted(set(val_targets))
    present_names = [REVERSE_LABEL_MAP[i] for i in present_labels]
    print(classification_report(
        val_targets, val_preds, labels=present_labels,
        target_names=present_names, digits=4
    ))

    # Confusion matrix
    print(f"{'─' * 40}")
    print("  CONFUSION MATRIX (Validation)")
    print(f"{'─' * 40}")
    cm = confusion_matrix(val_targets, val_preds, labels=present_labels)
    header = "  Predicted →  " + "  ".join(f"{n:>8}" for n in present_names)
    print(header)
    print("  " + "─" * len(header))
    for i, row_label in enumerate(present_names):
        row = "  ".join(f"{v:>8}" for v in cm[i])
        print(f"  {row_label:>10} │ {row}")

    # Recommendations
    print(f"\n{'─' * 40}")
    print("  RECOMMENDATIONS")
    print(f"{'─' * 40}")
    gap = train_f1 - val_f1
    if gap > 0.05:
        print("  ⚠ Train-Val F1 gap > 0.05 — model is overfitting.")
        print("    → Collect more data, increase dropout, or reduce model size.")
    elif gap < 0.01:
        print("  ⚠ Train-Val F1 gap < 0.01 — model may be underfitting.")
        print("    → Train longer, increase model capacity, or reduce regularization.")
    else:
        print("  ✓ Train-Val gap looks healthy.")

    if best_epoch < num_epochs * 0.4:
        print(f"  ⚠ Best epoch was early ({best_epoch}/{num_epochs}) — model peaked fast.")
        print("    → Lower learning rate or increase patience.")

    if best_epoch > num_epochs * 0.95:
        print(f"  ⚠ Best epoch was near the end ({best_epoch}/{num_epochs}).")
        print("    → Training more epochs may improve results.")

    report = classification_report(
        val_targets, val_preds, labels=present_labels,
        target_names=present_names, output_dict=True
    )
    for cls_name in present_names:
        if cls_name in report and report[cls_name]["f1-score"] < 0.85:
            print(f"  ⚠ {cls_name} F1 is low ({report[cls_name]['f1-score']:.4f}).")
            if report[cls_name]["recall"] < report[cls_name]["precision"]:
                print(f"    → Low recall — model misses {cls_name} strokes. Add more {cls_name} training data.")
            else:
                print(f"    → Low precision — model over-predicts {cls_name}. Check for label noise.")

    print("\n" + "=" * 60)
    print(f"  Model saved to '{model_save_path}'")
    print("=" * 60)
