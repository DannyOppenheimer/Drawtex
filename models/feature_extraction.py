import numpy as np
from scipy.stats import mode


# Number of features produced per stroke by extract_features() and
# extract_features_from_vectors(). Single source of truth — imported by
# the LSTM training, verification, and inference code.
INPUT_DIM = 10


def smooth_predictions(preds, window_size=5):
    """
    Applies a majority vote sliding window to smooth predictions.
    window_size must be odd.
    """
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
    """
    Extracts features and labels from a session dict with 'strokes' key.
    Each stroke has 'points' (list of [x, y, t]) and 'label'.
    """
    strokes = session_data.get("strokes", [])
    features = []
    labels = []

    LABEL_MAP = {"text": 0, "math": 1, "diagram": 2}

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
        aspect_ratio = width / (height + 1e-6)

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
            aspect_ratio,
        ]
        features.append(stroke_vec)
        labels.append(LABEL_MAP.get(stroke.get("label", "text"), 0))

        prev_end_x = xs[-1]
        prev_end_y = ys[-1]
        prev_end_t = ts[-1]

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def pad_sequence(batch):
    """Collate function for DataLoader with sequence padding."""
    import torch

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
