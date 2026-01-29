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
from sklearn.metrics import f1_score
import random
import joblib

from models.lstm import StrokeClassifierLSTM
from models.feature_extraction import extract_features, pad_sequence, smooth_predictions

LABEL_MAP = {"text": 0, "math": 1, "diagram": 2}
INPUT_DIM = 9
HIDDEN_DIM = 64
OUTPUT_DIM = 3
BATCH_SIZE = 32

DATA_PATH = str(PROJECT_ROOT / "data" / "collected_strokes.json")
WEIGHTS_DIR = PROJECT_ROOT / "weights"
SCALER_PATH = str(WEIGHTS_DIR / "scaler.save")
MODEL_SAVE_PATH = str(WEIGHTS_DIR / "best_model.pth")


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

    return f1_score(all_targets, all_preds, average="weighted")


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

    num_epochs = 75
    best_f1 = 0.0

    model = StrokeClassifierLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

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
            optimizer.step()
            train_loss += loss.item()

        # The evaluate function now returns the F1 of the SMOOTHED predictions
        val_f1 = evaluate(model, val_loader, device)
        avg_loss = train_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val F1 (Smoothed): {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved (F1: {best_f1:.4f})")

    print(f"\nTraining Complete. Best F1 Score: {best_f1:.4f}")
    print(f"Model saved to '{MODEL_SAVE_PATH}'")
