import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --- Constants (Must match training) ---
WINDOW_SIZE = 50
STRIDE = 2
BATCH_SIZE = 32
HIDDEN_DIM = 128
INPUT_DIM = 7  # Updated for new features
LABEL_MAP = {"text": 0, "math": 1, "diagram": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# --- Model Definition ---
class StrokeBiLSTM(nn.Module):
    def __init__(self):
        super(StrokeBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(HIDDEN_DIM * 2, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


# --- Feature Extraction (Robust Version) ---
def extract_features(points, prev_end_point):
    if not points or len(points) < 2:
        return np.zeros(INPUT_DIM, dtype=np.float32)

    pts = np.array([p[:2] for p in points])
    times = np.array([p[2] for p in points])

    deltas = np.diff(pts, axis=0)
    segment_dists = np.sqrt((deltas**2).sum(axis=1))

    start_x, start_y = pts[0]
    path_len = np.sum(segment_dists)

    # Robust Width/Height
    width = np.clip((np.max(pts[:, 0]) - np.min(pts[:, 0])) / 100.0, 0, 5.0)
    height = np.clip((np.max(pts[:, 1]) - np.min(pts[:, 1])) / 100.0, 0, 5.0)

    straight_dist = np.linalg.norm(pts[-1] - pts[0])
    linearity = straight_dist / (path_len + 1e-6)

    if prev_end_point is not None:
        p_x, p_y, p_t = prev_end_point
        # Robust dx/dy
        dx = (start_x - p_x) / 100.0
        dy = (start_y - p_y) / 100.0
        dx = np.clip(dx, -3.0, 3.0)
        dy = np.clip(dy, -3.0, 3.0)

        dt = times[0] - p_t
        if dt > 5.0:
            dt = 5.0
        if dt < 0.01:
            dt = 0.01
    else:
        dx, dy, dt = 0.0, 0.0, 0.0

    angle = np.arctan2(dy, dx)
    feat_sin = np.sin(angle)
    feat_cos = np.cos(angle)

    # New Features
    raw_dist = np.linalg.norm(
        [
            start_x - (prev_end_point[0] if prev_end_point else start_x),
            start_y - (prev_end_point[1] if prev_end_point else start_y),
        ]
    )
    speed = raw_dist / (dt + 1e-5)
    log_speed = np.log1p(speed)

    segment_angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    angle_changes = np.diff(segment_angles)
    angle_changes = np.arctan2(np.sin(angle_changes), np.cos(angle_changes))
    total_curvature = np.sum(np.abs(angle_changes))
    total_curvature = np.clip(total_curvature / 10.0, 0, 5.0)

    log_len = np.log1p(path_len)
    log_len = np.clip(log_len / 5.0, 0, 2.0)

    return np.array(
        [
            dx,
            dy,
            dt,
            width,
            height,
            linearity,
            path_len / 100.0,
            feat_sin,
            feat_cos,
            log_speed,
            total_curvature,
            log_len,
        ],
        dtype=np.float32,
    )


def prepare_windows(sessions):
    all_windows_X = []
    all_windows_Y = []
    continuous_feats = []
    continuous_labels = []

    for session in sessions:
        prev_end = None
        for stroke in session["strokes"]:
            label_id = LABEL_MAP.get(stroke.get("label", "text"), 0)
            raw_points = stroke["points"]
            feat = extract_features(raw_points, prev_end)
            continuous_feats.append(feat)
            continuous_labels.append(label_id)
            if raw_points:
                last_p = raw_points[-1]
                prev_end = (last_p[0], last_p[1], last_p[2])

    num_total_strokes = len(continuous_feats)
    for i in range(0, num_total_strokes, STRIDE):
        end_idx = i + WINDOW_SIZE
        if end_idx <= num_total_strokes:
            window_x = continuous_feats[i:end_idx]
            window_y = continuous_labels[i:end_idx]
            all_windows_X.append(window_x)
            all_windows_Y.append(window_y)
        else:
            window_x = continuous_feats[i:end_idx]
            window_y = continuous_labels[i:end_idx]
            current_len = len(window_x)
            if current_len > 0:
                pad_len = WINDOW_SIZE - current_len
                padding_x = [
                    np.zeros(INPUT_DIM, dtype=np.float32) for _ in range(pad_len)
                ]
                window_x.extend(padding_x)
                padding_y = [-1 for _ in range(pad_len)]
                window_y.extend(padding_y)
                all_windows_X.append(window_x)
                all_windows_Y.append(window_y)

    return np.array(all_windows_X), np.array(all_windows_Y)


# --- Main Evaluation Loop ---
if __name__ == "__main__":
    validation_file = "validation_data.json"
    model_path = "0.8738_stroke_lstm_best.pth"

    # 1. Load Data
    if not os.path.exists(validation_file):
        print(f"Error: {validation_file} not found.")
        exit()

    print(f"Loading {validation_file}...")
    with open(validation_file, "r") as f:
        val_sessions = json.load(f)

    # 2. Process Data
    print("Processing windows...")
    X_val, Y_val = prepare_windows(val_sessions)
    print(f"Generated {len(X_val)} windows.")

    # 3. Setup DataLoader
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.long),
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrokeBiLSTM().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Error: Model file not found.")
        exit()

    # 5. Inference
    model.eval()
    all_preds = []
    all_targets = []

    print("Running inference...")
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)

            # Reshape for classification
            flat_out = out.view(-1, 3)
            flat_y = by.view(-1)

            # Mask padding (-1)
            mask = flat_y != -1

            preds = torch.argmax(flat_out, dim=1)

            all_preds.extend(preds[mask].cpu().numpy())
            all_targets.extend(flat_y[mask].cpu().numpy())

    # 6. Report
    print("\n" + "=" * 30)
    print("FINAL VALIDATION RESULTS")
    print("=" * 30)

    acc = accuracy_score(all_targets, all_preds)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(
        classification_report(
            all_targets, all_preds, target_names=list(LABEL_MAP.keys())
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
