import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import random
import math
import copy
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
WINDOW_SIZE = 50  # Sequence length
STRIDE = 2  # Slide window 2 strokes at a time (High Density)
AUGMENT_COPIES = 5  # Generate 5 fake variations for every real session
BATCH_SIZE = 32  # Increased batch size for larger data
HIDDEN_DIM = 128
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10
MAX_ERRORS_TO_SAVE = 20

# Label Mapping
LABEL_MAP = {"text": 0, "math": 1, "diagram": 2}


def apply_smoothing(predictions, kernel_size=5):
    """
    Replaces each prediction with the median of its neighbors.
    Removes random noise spikes.
    """
    # Kernel size must be odd (3, 5, 7)
    return medfilt(predictions, kernel_size=kernel_size)


# --- 1. AUGMENTATION ENGINE ---
def augment_stroke(points):
    """
    Applies geometric distortions: Scale, Rotate, Jitter, and Rushed Writing.
    """
    if not points or len(points) < 2:
        return points

    # Convert to numpy
    arr = np.array(points, dtype=np.float32)
    coords = arr[:, :2]  # x, y
    times = arr[:, 2]  # t

    # A. Rushed Writing (Downsampling)
    # Randomly drop 20-30% of points to simulate fast, jagged writing
    if len(points) > 15 and random.random() < 0.5:
        # Keep start/end anchors
        indices = (
            [0]
            + sorted(random.sample(range(1, len(points) - 1), int(len(points) * 0.75)))
            + [len(points) - 1]
        )
        coords = coords[indices]
        times = times[indices]

    # B. Scaling (Zoom)
    scale = random.uniform(0.85, 1.15)
    coords = coords * scale

    # C. Rotation (Tilt)
    angle_deg = random.uniform(-10, 10)
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))

    center = coords.mean(axis=0)
    coords = (coords - center).dot(rotation_matrix) + center

    # D. Jitter (Sensor Noise)
    noise = np.random.normal(0, 0.5, coords.shape)
    coords = coords + noise

    # E. Time Warping
    time_scale = random.uniform(0.8, 1.2)
    times = times * time_scale

    # Recombine
    new_points = np.column_stack((coords, times))
    return new_points.tolist()


# --- 2. FEATURE EXTRACTION ---
def extract_features(points, prev_end_point):
    """
    Converts raw points [x,y,t] into 7-dim vector.
    """
    if not points or len(points) < 2:
        return np.zeros(7, dtype=np.float32)

    pts = np.array([p[:2] for p in points])
    times = np.array([p[2] for p in points])

    start_x, start_y = pts[0]

    # Geometric Stats
    deltas = np.diff(pts, axis=0)
    segment_lengths = np.sqrt((deltas**2).sum(axis=1))
    path_len = np.sum(segment_lengths)

    width = np.max(pts[:, 0]) - np.min(pts[:, 0])
    height = np.max(pts[:, 1]) - np.min(pts[:, 1])

    straight_dist = np.linalg.norm(pts[-1] - pts[0])
    linearity = straight_dist / (path_len + 1e-6)

    # Context (Relation to previous stroke)
    if prev_end_point is not None:
        p_x, p_y, p_t = prev_end_point
        dx = start_x - p_x
        dy = start_y - p_y
        dt = times[0] - p_t
        if dt > 5.0:
            dt = 5.0  # Cap pauses
    else:
        dx, dy, dt = 0.0, 0.0, 0.0

    angle = np.arctan2(dy, dx)
    feat_sin = np.sin(angle)
    feat_cos = np.cos(angle)

    # Normalized Features
    return np.array(
        [
            dx / 100.0,
            dy / 100.0,
            dt,
            width / 100.0,
            height / 100.0,
            linearity,
            path_len / 100.0,
            feat_sin,
            feat_cos,
        ],
        dtype=np.float32,
    )


# --- 3. DATASET GENERATION ---
def prepare_windows(sessions, augment=False):
    all_windows_X = []
    all_windows_Y = []

    # 1. Collect ALL features into one giant stream
    # We treat all sessions as one long continuous "day of writing"
    continuous_feats = []
    continuous_labels = []

    for session in sessions:
        prev_end = None  # Reset context at start of each real session

        for stroke in session["strokes"]:
            label_id = LABEL_MAP.get(stroke.get("label", "text"), 0)
            raw_points = stroke["points"]

            if augment:
                raw_points = augment_stroke(raw_points)

            feat = extract_features(raw_points, prev_end)

            continuous_feats.append(feat)
            continuous_labels.append(label_id)

            if raw_points:
                last_p = raw_points[-1]
                prev_end = (last_p[0], last_p[1], last_p[2])

    # 2. Slice the Giant Stream
    # Now we have one list with 50,000 strokes. We slide over IT.
    num_total_strokes = len(continuous_feats)

    for i in range(0, num_total_strokes, STRIDE):
        end_idx = i + WINDOW_SIZE

        # If we have enough data left for a full window, take it
        if end_idx <= num_total_strokes:
            window_x = continuous_feats[i:end_idx]
            window_y = continuous_labels[i:end_idx]

            all_windows_X.append(window_x)
            all_windows_Y.append(window_y)

        # If we are at the very end of the giant stream, we can drop the last few
        # (Padding is less critical now that we have thousands of full windows)

    return np.array(all_windows_X), np.array(all_windows_Y)


# --- 4. MODEL ---
class StrokeBiLSTM(nn.Module):
    def __init__(self):
        super(StrokeBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=7,
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# --- 5. MAIN TRAINING LOOP ---
if __name__ == "__main__":

    # 1. Load Data
    try:
        with open("collected_strokes.json", "r") as f:
            raw_sessions = json.load(f)
    except:
        print("Error: 'collected_strokes.json' not found.")
        raw_sessions = []  # Empty for safety

    if raw_sessions:
        print(f"Total Raw Sessions: {len(raw_sessions)}")

        # 2. SPLIT SESSIONS (Prevent Data Leakage)
        # We split raw sessions FIRST, so Training Windows and Test Windows
        # never come from the same drawing session.
        train_sessions, val_sessions = train_test_split(
            raw_sessions, test_size=0.2, random_state=42
        )

        print(f"Training Sessions: {len(train_sessions)}")
        print(f"Validation Sessions: {len(val_sessions)}")

        # 3. Generate Training Data (Clean + Augmented)
        print("\nGenerating Training Windows...")
        X_train_clean, Y_train_clean = prepare_windows(train_sessions, augment=False)

        train_X_list = [X_train_clean]
        train_Y_list = [Y_train_clean]

        print(f"   Clean Windows: {len(X_train_clean)}")

        # Augmentation Loop
        for i in range(AUGMENT_COPIES):
            print(f"   Augmenting pass {i+1}/{AUGMENT_COPIES}...")
            X_aug, Y_aug = prepare_windows(train_sessions, augment=True)
            train_X_list.append(X_aug)
            train_Y_list.append(Y_aug)

        X_train = np.concatenate(train_X_list, axis=0)
        Y_train = np.concatenate(train_Y_list, axis=0)
        print(f"Total Training Samples: {len(X_train)}")

        # 4. Generate Validation Data (Clean Only - No Augmentation for Val)
        print("\nGenerating Validation Windows...")
        X_val, Y_val = prepare_windows(val_sessions, augment=False)
        print(f"Total Validation Samples: {len(X_val)}")

        # 5. Create Loaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.long),
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 6. Setup Training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nTraining on {device}...")

        model = StrokeBiLSTM().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = FocalLoss(gamma=2)  # Ignore padding

        best_model_wts = None
        best_val_f1 = 0.0

        epochs_no_improvement = 0

        for epoch in range(EPOCHS):
            # --- TRAIN ---
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out.view(-1, 3), by.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # --- VALIDATE ---
            model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    loss = criterion(out.view(-1, 3), by.view(-1))
                    val_loss += loss.item()

                    # Metrics
                    flat_out = out.view(-1, 3)
                    flat_y = by.view(-1)
                    mask = flat_y != -1

                    preds = torch.argmax(flat_out, dim=1)
                    all_preds.extend(preds[mask].cpu().numpy())
                    all_targets.extend(flat_y[mask].cpu().numpy())

            # Stats
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_f1_macro = f1_score(all_targets, all_preds, average="macro")

            print(
                f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1 (Macro): {val_f1_macro:.4f}"
            )

            # Save Best

            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"Best F1: {best_val_f1:.4f} | Saving model.")
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
                print(f"  No improvement for {epochs_no_improvement} epochs.")

            if epochs_no_improvement >= 10:
                break

        # Save Final Best Model
        if best_model_wts:
            torch.save(best_model_wts, "stroke_lstm_best.pth")
            print(f"\nBest model saved with Val F1: {best_val_f1:.4f}")
