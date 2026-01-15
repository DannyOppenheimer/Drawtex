import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import mode  # <--- Added for majority voting
import random

LABEL_MAP = {"text": 0, "math": 1, "diagram": 2}
INPUT_DIM = 9
HIDDEN_DIM = 64
OUTPUT_DIM = 3
BATCH_SIZE = 32


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(67)


def smooth_predictions(preds, window_size=5):
    """
    Applies a majority vote sliding window to smooth predictions.
    window_size must be odd.
    """
    if len(preds) < window_size:
        return preds

    smoothed = np.copy(preds)
    half_window = window_size // 2

    # We iterate over the sequence and take the mode of the window
    for i in range(len(preds)):
        start = max(0, i - half_window)
        end = min(len(preds), i + half_window + 1)
        window = preds[start:end]

        # mode returns (mode_array, count_array), we want the first element of mode_array
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

        linearity = euclidean_dist / path_length  # 1.0 = straight line
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
    # Sort by length (required for pack_padded_sequence)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    features = [torch.tensor(x[0]) for x in batch]
    labels = [torch.tensor(x[1]) for x in batch]

    # Capture the actual lengths
    lengths = torch.tensor([len(f) for f in features])

    features_padded = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-1
    )

    return features_padded, labels_padded, lengths


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
        # PACK the sequence
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # Run LSTM on only the real data
        packed_out, _ = self.lstm(packed_x)

        # UNPACK so we can run the linear layer
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        logits = self.fc(lstm_out)
        return logits


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

    try:
        with open("collected_strokes.json", "r") as f:
            raw_sessions = json.load(f)
    except FileNotFoundError:
        print("Error: collected_strokes.json not found.")
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

    import joblib

    joblib.dump(scaler, "scaler.save")  # <--- ADD THIS
    print("Scaler saved to 'scaler.save'")

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
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> New best model saved (F1: {best_f1:.4f})")

    print(f"\nTraining Complete. Best F1 Score: {best_f1:.4f}")
    print("Model saved to 'best_model.pth'")
