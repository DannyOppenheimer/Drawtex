import json
import sys
from pathlib import Path

# Project root directory (Drawtex/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

from models.lstm import StrokeClassifierLSTM
from models.feature_extraction import (
    INPUT_DIM,
    extract_features,
    pad_sequence,
    smooth_predictions,
)

# ==========================================
# CONFIGURATION (Must match training script)
# ==========================================
MODEL_PATH = str(PROJECT_ROOT / "weights" / "0_9163_best_model.pth")
TEST_FILE = str(PROJECT_ROOT / "data" / "validation_data.json")
LABEL_MAP = {"text": 0, "math": 1, "diagram": 2}
REVERSE_LABEL_MAP = {0: "text", 1: "math", 2: "diagram"}
HIDDEN_DIM = 64
OUTPUT_DIM = 3
BATCH_SIZE = 1  # Inference is usually done 1 doc at a time, or small batches

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        with open(TEST_FILE, "r") as f:
            raw_sessions = json.load(f)
        print(f"Loaded {len(raw_sessions)} sessions from {TEST_FILE}")
    except FileNotFoundError:
        print(f"Error: {TEST_FILE} not found.")
        exit()

    X_list = []
    Y_list = []

    # Keep track of original session indices if needed for debugging
    for session in raw_sessions:
        feats, labs = extract_features(session)
        if len(feats) > 0:
            X_list.append(feats)
            Y_list.append(labs)

    if not X_list:
        print("No valid stroke data found.")
        exit()

    # 2. Normalize (Fit Scaler on this batch - see note in intro)
    all_features = np.vstack(X_list)
    scaler = StandardScaler()
    scaler.fit(all_features)
    X_list_normalized = [scaler.transform(x) for x in X_list]

    # 3. Prepare Loader
    data = list(zip(X_list_normalized, Y_list))
    loader = DataLoader(
        data, batch_size=BATCH_SIZE, collate_fn=pad_sequence, shuffle=False
    )

    # 4. Load Model
    model = StrokeClassifierLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        exit()

    # 5. Inference Loop
    model.eval()
    all_preds = []
    all_targets = []

    print("\nRunning Inference...")

    with torch.no_grad():
        for batch_i, (batch_x, batch_y, batch_lengths) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x, batch_lengths)
            preds = torch.argmax(logits, dim=2)

            # Process per document in batch
            for i in range(len(batch_y)):
                length = batch_lengths[i].item()

                # Get raw sequences
                raw_pred = preds[i, :length].cpu().numpy()
                target = batch_y[i, :length].cpu().numpy()

                # Apply Smoothing
                smoothed_pred = smooth_predictions(raw_pred, window_size=5)

                all_preds.extend(smoothed_pred)
                all_targets.extend(target)

    # 6. Final Report
    score = f1_score(all_targets, all_preds, average="weighted")
    print("-" * 30)
    print(f"Weighted F1 Score: {score:.4f}")
    print("-" * 30)
    print("\nDetailed Report:")
    print(
        classification_report(
            all_targets, all_preds, target_names=["Text", "Math", "Diagram"]
        )
    )
