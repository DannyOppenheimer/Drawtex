import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pix2tex.cli import LatexOCR
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
import warnings
import os
import uuid
import joblib
from scipy.interpolate import make_interp_spline, splprep, splev

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# ==========================================
# 1. CONFIGURATION & MODEL DEFINITIONS
# ==========================================
LSTM_MODEL_PATH = "0_9163_best_model.pth"
INPUT_DIM = 9  # Must match your trained model
HIDDEN_DIM = 64
OUTPUT_DIM = 3
LABEL_MAP = {0: "text", 1: "math", 2: "diagram"}


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


# ==========================================
# 2. FEATURE EXTRACTION & UTILS
# ==========================================
def extract_features_from_vectors(vectors):
    """
    Converts raw list of vectors [[(x,y,t)...], ...] into LSTM features.
    """
    features = []

    # We need to simulate the "prev" variables across the session
    if not vectors:
        return np.array([], dtype=np.float32)

    prev_end_x = 0
    prev_end_y = 0
    prev_end_t = vectors[0][0][2] if vectors and vectors[0] else 0

    for points_list in vectors:
        points = np.array(points_list)
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

        prev_end_x = xs[-1]
        prev_end_y = ys[-1]
        prev_end_t = ts[-1]

    return np.array(features, dtype=np.float32)


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


def save_debug_image(img_obj, label="unknown"):
    """
    Saves a PIL Image or Numpy Array to disk with a unique ID.
    """
    DEBUG_FOLDER = "debug_images"
    if not os.path.exists(DEBUG_FOLDER):
        os.makedirs(DEBUG_FOLDER)

    # Generate a unique filename: debug_images/math_a1b2c3d4.png
    filename = f"{label}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(DEBUG_FOLDER, filename)

    try:
        if isinstance(img_obj, Image.Image):
            # It's a PIL Image
            img_obj.save(filepath)
        elif isinstance(img_obj, np.ndarray):
            # It's a Numpy Array (OpenCV)
            # Check if it needs normalization (0-1 floats vs 0-255 ints)
            if img_obj.max() <= 1.0:
                img_obj = (img_obj * 255).astype(np.uint8)
            cv2.imwrite(filepath, img_obj)

        print(f" -> Debug saved: {filepath}")
    except Exception as e:
        print(f" ! Failed to save debug image: {e}")


def smooth_points(points_nx2, num_points_factor=5):
    """
    Takes an Nx2 numpy array of (x,y) points and uses B-spline approximation
    to generate a smoother path.
    Includes duplicate point filtering and a smoothing factor to prevent "teeth".
    """
    # Need at least 3 points to define a curve
    if len(points_nx2) < 3:
        return points_nx2

    # --- FILTER DUPLICATES (Keep this from before) ---
    diffs = np.diff(points_nx2, axis=0)
    dist_sq = np.sum(diffs**2, axis=1)
    # Keep the first point, plus any point that is effectively different
    mask = np.concatenate([[True], dist_sq > 1e-6])
    points_nx2 = points_nx2[mask]
    # -------------------------

    # Re-check length after filtering.
    # splprep needs at least k+1 points.
    num_points = len(points_nx2)
    if num_points < 4:  # Need 4 points for cubic spline (k=3)
        return points_nx2

    # Degree of the spline. Cubic (3) is standard for smoothness.
    k = 3

    try:
        # --- THE FIX IS HERE ---
        # Use splprep for parametric spline fitting.
        # 's' is the smoothing factor.
        # s=0 means pass through every point (what you had before).
        # Larger 's' means more smoothing.
        # Heuristic: s = num_points * constant usually works well.
        smoothing_factor = num_points * 3.0

        # tck is the tuple (knots, coefficients, degree)
        # splprep expects points as a list of arrays like [x_array, y_array]
        tck, u = splprep(points_nx2.T, s=smoothing_factor, k=k)

        # Evaluate the spline over a denser grid for rendering
        u_dense = np.linspace(u.min(), u.max(), num_points * num_points_factor)
        x_dense, y_dense = splev(u_dense, tck)

        return np.stack([x_dense, y_dense], axis=1)

    except Exception as e:
        # Fallback to raw points if smoothing fails for some reason
        # print(f"Smoothing Warning (Skipping): {e}") # Uncomment for debugging
        return points_nx2


def render_strokes_to_image(strokes, label_type="text", padding=40):
    """
    Renders strokes with configuration dependent on the label type (math vs text),
    and applies smoothing.
    """
    # Extract just XY coordinates for rendering and calculate bounds
    all_points_xy = []
    all_x_raw, all_y_raw = [], []

    for s in strokes:
        pts = np.array(s)
        if len(pts) > 0:
            xy_only = pts[:, :2]  # Drop time dimension for rendering
            all_points_xy.append(xy_only)
            all_x_raw.extend(xy_only[:, 0])
            all_y_raw.extend(xy_only[:, 1])

    if not all_x_raw:
        return None

    # 1. Determine Bounds
    min_x, max_x = min(all_x_raw), max(all_x_raw)
    min_y, max_y = min(all_y_raw), max(all_y_raw)
    raw_width = max_x - min_x
    raw_height = max_y - min_y

    # Ensure no zero-dimension images
    if raw_height < 1:
        raw_height = 1
    if raw_width < 1:
        raw_width = 1

    # =======================================================
    # 2. Branching Configuration based on Type
    # =======================================================
    if label_type == "math":
        # Math needs to be taller and bolder for pix2tex ViT
        target_height = 256
        # Thickness is ~3.5% of the height
        stroke_width_ratio = 0.035
        min_stroke_width = 5
    else:  # "text"
        # Text should be standard handwriting height, thinner lines so loops don't blob
        target_height = 128
        # Thickness is ~1.5% of the height
        stroke_width_ratio = 0.015
        min_stroke_width = 3
    # =======================================================

    # 3. Calculate Scaling
    scale = target_height / raw_height
    new_width = int(raw_width * scale)
    new_height = int(raw_height * scale)

    # Calculate dynamic stroke width
    stroke_width = max(min_stroke_width, int(target_height * stroke_width_ratio))

    # Create Canvas
    img_w = new_width + 2 * padding
    img_h = new_height + 2 * padding
    image = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(image)

    # 4. Render Loop (Smoothing -> Scaling -> Drawing)
    for raw_pts_nx2 in all_points_xy:
        if len(raw_pts_nx2) < 2:
            continue

        # A) SMOOTHING
        # Apply spline interpolation to reduce choppiness
        pts_to_render = smooth_points(raw_pts_nx2)

        # B) SCALING & SHIFTING
        xy_tuples = []
        for x, y in zip(pts_to_render[:, 0], pts_to_render[:, 1]):
            new_x = (x - min_x) * scale + padding
            new_y = (y - min_y) * scale + padding
            xy_tuples.append((new_x, new_y))

        # C) DRAWING
        # Draw line
        draw.line(xy_tuples, fill="black", width=stroke_width)

        # Add round caps to joints to make it look like a fluid marker stroke
        # instead of connected rectangles.
        r = stroke_width / 2
        for x, y in xy_tuples:
            draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    print(
        f"DEBUG: Rendered [{label_type}] Size: {img_w}x{img_h}, Stroke: {stroke_width}px"
    )
    save_debug_image(image, label=f"Rendered_{label_type}")

    return image


# ==========================================
# 3. GLOBAL MODEL LOADERS (Singleton pattern)
# ==========================================
_lstm_model = None
_text_processor = None
_text_model = None
_math_model = None


def load_models():
    global _lstm_model, _text_processor, _text_model, _math_model

    print("--- Loading Models (Local CPU Optimization) ---")

    # 1. LSTM Classifier
    _lstm_model = StrokeClassifierLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    try:
        # Load CPU weights
        _lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location="cpu"))
        _lstm_model.eval()
        print(" [x] LSTM Classifier loaded.")
    except FileNotFoundError:
        print(f" [!] Error: {LSTM_MODEL_PATH} not found.")
        return False

    # 2. TrOCR (Text) - QUANTIZED
    _text_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    _text_model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten"
    )

    # 3. Pix2Tex (Math)
    print(" ... Loading LatexOCR...")
    _math_model = LatexOCR()
    print(" [x] LatexOCR loaded.")

    return True


# ==========================================
# 4. MAIN ANALYZE FUNCTION
# ==========================================
def analyze_vectors(vectors):
    """
    Main entry point. Receives list of list of (x,y,t).
    Prints final LaTeX string.
    """
    if not vectors:
        return

    # A. Load models if not loaded
    if _lstm_model is None:
        if not load_models():
            return

    # B. Feature Extraction & Normalization
    print(f"\nAnalyzing {len(vectors)} strokes...")
    features = extract_features_from_vectors(vectors)

    # Sanity check: If document is empty, return early
    if len(features) == 0:
        return

    # C. Normalization (Load the training scaler)
    try:
        scaler = joblib.load("scaler.save")
        # print("Loaded scaler from training.") # Optional logging
    except FileNotFoundError:
        print(
            "Error: scaler.save not found. You must run the training script once to generate it."
        )
        return

    # CRITICAL FIX: Transform the features using the loaded scaler.
    # Do NOT use fit_transform (that would erase the "global" size knowledge).
    features_norm = scaler.transform(features)

    # D. Classification (LSTM)
    # Convert to Tensor (Explicitly Float32 to match LSTM weights)
    x_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(
        0
    )  # [1, Seq_Len, Features]
    lengths = torch.tensor([len(features_norm)])

    with torch.no_grad():
        logits = _lstm_model(x_tensor, lengths)
        preds = torch.argmax(logits, dim=2).squeeze(0).numpy()

    # D. Smoothing
    smoothed_preds = smooth_predictions(preds, window_size=5)
    print(f"Predictions: {smoothed_preds}")

    # E. Grouping & OCR Routing
    full_latex = ""

    current_label = smoothed_preds[0]
    current_strokes = []

    # Helper to flush a group
    def process_group(label_idx, stroke_group):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not stroke_group:
            return ""

        label_name = LABEL_MAP.get(label_idx, "text")

        # Render
        img = render_strokes_to_image(stroke_group, label_type=label_name)
        if img is None:
            return ""

        result = ""

        if label_name == "math":
            print("   -> Processing Math block...")
            try:
                # LatexOCR handles its own resizing
                latex = _math_model(img)

                result = f"\n$${latex}$$\n"
            except Exception as e:
                print(f"Math Error: {e}")

        elif label_name == "text":
            print("   -> Processing Text block...")

            # --- SANITIZATION STEP ---
            # 1. Handle NumPy Arrays
            if isinstance(img, np.ndarray):
                # If float (0.0 - 1.0), scale up to 255
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                # Convert to PIL
                img = Image.fromarray(img)

            # 2. Force RGB (Processors hate Grayscale/Alpha channels)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # -------------------------

            _text_model.to(device)
            pixel_values = _text_processor(
                images=img, return_tensors="pt"
            ).pixel_values.to(
                device
            )  # Don't forget .to(device)

            # Ensure model is in eval mode!
            _text_model.eval()

            generated_ids = _text_model.generate(pixel_values)
            text = _text_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            result = f"{text} "

        elif label_name == "diagram":
            print("   -> Skipping Diagram (not implemented yet)")
            result = "\n[DIAGRAM PLACEHOLDER]\n"

        return result

    # Loop through strokes
    for i, (vec, label) in enumerate(zip(vectors, smoothed_preds)):
        if label == current_label:
            current_strokes.append(vec)
        else:
            # Process previous group
            full_latex += process_group(current_label, current_strokes)
            # Start new
            current_label = label
            current_strokes = [vec]

    # Process final group
    full_latex += process_group(current_label, current_strokes)

    print("\n" + "=" * 40)
    print("FINAL DOCUMENT SOURCE:")
    print("=" * 40)
    print(full_latex)
    print("=" * 40)
