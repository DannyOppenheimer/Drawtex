import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from scipy.stats import mode

# ==========================================
# CONFIGURATION (Must match training script)
# ==========================================
MODEL_PATH = "0_9163_best_model.pth"
TEST_FILE = "validation_data.json"
LABEL_MAP = {"text": 0, "math": 1, "diagram": 2}
REVERSE_LABEL_MAP = {0: "text", 1: "math", 2: "diagram"}
INPUT_DIM = 9
HIDDEN_DIM = 64
OUTPUT_DIM = 3
BATCH_SIZE = 1  # Inference is usually done 1 doc at a time, or small batches

# ==========================================
# CLASSES & FUNCTIONS
# ==========================================


class StrokeClassifierLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StrokeClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed_x)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )
        logits = self.fc(lstm_out)
        return logits


def smooth_predictions(preds, window_size=5):
    if len(preds) < window_size:
        return preds
    smoothed = np.copy(preds)
    half_window = window_size // 2
    for i in range(len(preds)):
        start = max(0, i - half_window)
        end = min(len(preds), i + half_window + 1)
        window = preds[start:end]
        val = mode(window, keepdims=False)[0]
        smoothed[i] = val
    return smoothed


def extract_features(session_data):
    strokes = session_data.get("strokes", [])
    features = []
    labels = []

    prev_end_x = 0
    prev_end_y = 0
    prev_end_t = strokes[0]["points"][0][2] if strokes and strokes[0]["points"] else 0

    for stroke in strokes:
        points = np.array(stroke["points"])
        if len(points) == 0:
            continue

        xs, ys, ts = points[:, 0], points[:, 1], points[:, 2]

        dx_prev = xs[0] - prev_end_x
        dy_prev = ys[0] - prev_end_y
        dt_prev = ts[0] - prev_end_t
        width = np.max(xs) - np.min(xs)
        height = np.max(ys) - np.min(ys)
        duration = ts[-1] - ts[0]
        num_points = len(points)

        path_length = np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)) + 1e-6
        euclidean_dist = np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)
        linearity = euclidean_dist / path_length
        speed = path_length / (duration + 1e-6)

        stroke_vec = [
            dx_prev,
            dy_prev,
            dt_prev,
            width,
            height,
            duration,
            num_points,
            linearity,
            speed,
        ]
        features.append(stroke_vec)
        labels.append(LABEL_MAP.get(stroke.get("label", "text"), 0))

        prev_end_x = xs[-1]
        prev_end_y = ys[-1]
        prev_end_t = ts[-1]

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def pad_sequence(batch):
    # Sort by length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    features = [torch.tensor(x[0]) for x in batch]
    labels = [torch.tensor(x[1]) for x in batch]
    lengths = torch.tensor([len(f) for f in features])

    features_padded = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-1
    )

    return features_padded, labels_padded, lengths


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
