import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
import warnings
import os
import sys
import uuid
import joblib
from pathlib import Path
from scipy.interpolate import splprep, splev

# Make project root importable when this module is loaded as core.analyze
_PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORT))

from models.lstm import StrokeClassifierLSTM
from models.feature_extraction import INPUT_DIM

# Try to import Pix2Text (preferred), fallback to pix2tex
try:
    from pix2text import Pix2Text
    USING_PIX2TEXT = True
except ImportError:
    try:
        from pix2tex.cli import LatexOCR
        USING_PIX2TEXT = False
    except ImportError:
        USING_PIX2TEXT = None
        print("[!] Warning: Neither pix2text nor pix2tex found. Math OCR will be unavailable.")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Project root directory (Drawtex/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================
# 1. CONFIGURATION & MODEL DEFINITIONS
# ==========================================
LSTM_MODEL_PATH = str(PROJECT_ROOT / "weights" / "0_9163_best_model.pth")
SCALER_PATH = str(PROJECT_ROOT / "weights" / "scaler.save")
DEBUG_FOLDER = str(PROJECT_ROOT / "debug_images")
HIDDEN_DIM = 64
OUTPUT_DIM = 3
LABEL_MAP = {0: "text", 1: "math", 2: "diagram"}


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
    prev_end_t = vectors[0][0][2] if vectors and vectors[0] and len(vectors[0][0]) >= 3 else 0

    for points_list in vectors:
        points = np.array(points_list)
        if len(points) == 0:
            continue

        # Validate that points have at least 3 dimensions (x, y, t)
        if points.ndim < 2 or points.shape[1] < 3:
            print(f"Warning: Stroke has invalid shape {points.shape}, expected (N, 3+)")
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


def split_math_groups_by_vertical_gap(strokes, gap_threshold=50):
    """
    Splits a list of strokes into separate groups based on vertical gaps.
    Uses vertical midpoint clustering so overlapping descenders/ascenders
    between lines don't prevent splitting.

    Args:
        strokes: List of stroke data [[(x,y,t), ...], ...]
        gap_threshold: Minimum vertical gap (pixels) to trigger a split.
                       If None, auto-detects based on median stroke height.

    Returns:
        List of stroke groups: [[[stroke1], [stroke2]], [[stroke3], [stroke4]], ...]
    """
    if not strokes:
        return []

    # Calculate bounding boxes for each stroke
    stroke_bounds = []
    for stroke in strokes:
        pts = np.array(stroke)
        if len(pts) > 0:
            min_y = np.min(pts[:, 1])
            max_y = np.max(pts[:, 1])
            mid_y = (min_y + max_y) / 2.0
            height = max_y - min_y
            stroke_bounds.append((min_y, max_y, mid_y, height, stroke))

    if not stroke_bounds:
        return []

    # Auto-detect threshold from median stroke height
    heights = [h for _, _, _, h, _ in stroke_bounds if h > 2]
    if heights:
        median_h = float(np.median(heights))
        # Lines are typically ~1.5-2x character height apart
        gap_threshold = max(gap_threshold, median_h * 0.8)

    # Sort by vertical midpoint (top to bottom)
    stroke_bounds.sort(key=lambda x: x[2])

    # Debug: print midpoints and threshold
    print(f"   [split] gap_threshold={gap_threshold:.1f}, stroke midpoints: "
          f"{[f'{m:.0f}' for _, _, m, _, _ in stroke_bounds]}")

    # Group strokes by comparing against the group's mean midpoint
    groups = []
    current_group_strokes = [stroke_bounds[0][4]]
    current_group_mids = [stroke_bounds[0][2]]

    for i in range(1, len(stroke_bounds)):
        min_y, max_y, mid_y, height, stroke = stroke_bounds[i]

        # Compare against the mean midpoint of the current group
        group_mean_mid = sum(current_group_mids) / len(current_group_mids)
        gap = mid_y - group_mean_mid

        if gap < gap_threshold:
            current_group_strokes.append(stroke)
            current_group_mids.append(mid_y)
        else:
            groups.append(current_group_strokes)
            current_group_strokes = [stroke]
            current_group_mids = [mid_y]

    groups.append(current_group_strokes)
    print(f"   [split] Result: {len(groups)} group(s)")

    return groups


def _format_list_item(text):
    """
    Post-processes OCR text to detect list formatting.
    Returns formatted LaTeX string.

    Detects:
      - Bullet lists: lines starting with -, *, •
      - Numbered lists: 1. / 2. / 1) / 2) / (1) / (2) etc.
    """
    import re

    # Bullet: starts with -, *, or •
    bullet_match = re.match(r'^[\-\*\u2022]\s*(.*)', text)
    if bullet_match:
        return f"\\item {bullet_match.group(1)}"

    # Numbered: (1) or (2) etc.
    paren_num_match = re.match(r'^\((\d+)\)\s*(.*)', text)
    if paren_num_match:
        return f"\\item[({paren_num_match.group(1)})] {paren_num_match.group(2)}"

    # Numbered: 1) or 2) etc.
    num_paren_match = re.match(r'^(\d+)\)\s*(.*)', text)
    if num_paren_match:
        return f"\\item[{num_paren_match.group(1)})] {num_paren_match.group(2)}"

    # Numbered: 1. or 2. etc.
    num_dot_match = re.match(r'^(\d+)\.\s*(.*)', text)
    if num_dot_match:
        return f"\\item[{num_dot_match.group(1)}.] {num_dot_match.group(2)}"

    return text


def save_debug_image(img_obj, label="unknown"):
    """
    Saves a PIL Image or Numpy Array to disk with a unique ID.
    """
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
            # It's a Numpy Array - convert to PIL to save
            # Check if it needs normalization (0-1 floats vs 0-255 ints)
            if img_obj.max() <= 1.0:
                img_obj = (img_obj * 255).astype(np.uint8)
            # Convert numpy array to PIL Image and save
            Image.fromarray(img_obj).save(filepath)

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
        # Reduced from 3.0 to 1.5 to prevent over-smoothing
        smoothing_factor = num_points * 1.5

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


def render_strokes_to_image(strokes, label_type="text", padding=None):
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
        min_stroke_width = 6
        # More padding for math to help pix2tex model
        if padding is None:
            padding = 60
    else:  # "text"
        # Text needs sufficient height and thickness for TrOCR recognition
        target_height = 192
        # Thickness is ~2.5% of the height for better character recognition
        stroke_width_ratio = 0.025
        min_stroke_width = 4
        # Standard padding for text
        if padding is None:
            padding = 40
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


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    global _lstm_model, _text_processor, _text_model, _math_model

    print(f"--- Loading Models (Device: {_device}) ---")

    # 1. LSTM Classifier
    _lstm_model = StrokeClassifierLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    try:
        # Load CPU weights
        _lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=_device))
        _lstm_model.to(_device)
        _lstm_model.eval()
        print(" [x] LSTM Classifier loaded.")
    except FileNotFoundError:
        print(f" [!] Error: {LSTM_MODEL_PATH} not found.")
        return False
    except (RuntimeError, OSError, EOFError) as e:
        print(f" [!] Error loading LSTM model: {e}")
        return False

    # 2. TrOCR (Text)
    try:
        print(" ... Loading TrOCR models...")
        _text_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _text_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-handwritten"
        ).to(_device)
        print(" [x] TrOCR models loaded.")
    except Exception as e:
        print(f" [!] Error loading TrOCR models: {e}")
        print("     Continuing without text OCR capability...")
        _text_processor = None
        _text_model = None

    # 3. Math OCR (Pix2Text preferred, fallback to pix2tex)
    try:
        if USING_PIX2TEXT is None:
            print(" [!] No math OCR library available (install pix2text or pix2tex)")
            _math_model = None
        elif USING_PIX2TEXT:
            print(" ... Loading Pix2Text (better accuracy)...")
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            ocr_device = 'cuda' if 'CUDAExecutionProvider' in providers else 'cpu'

            # Monkey-patch pix2text's download function to use the Python API
            # instead of subprocess huggingface-cli, which fails on Windows
            import pix2text.utils as _p2t_utils
            from huggingface_hub import snapshot_download as _hf_snapshot_download

            def _patched_hf_download(remote_repo, model_dir, env=None):
                endpoint = None
                if env and 'HF_ENDPOINT' in env:
                    endpoint = env['HF_ENDPOINT']
                _hf_snapshot_download(
                    repo_id=remote_repo,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                    endpoint=endpoint,
                )

            _p2t_utils.run_hf_download_cmd = _patched_hf_download

            from pix2text.latex_ocr import LatexOCR as P2TLatexOCR
            _math_model = P2TLatexOCR(device=ocr_device)
            print(" [x] Pix2Text LatexOCR loaded successfully.")
        else:
            print(" ... Loading LatexOCR (pix2tex fallback)...")
            _math_model = LatexOCR()
            print(" [x] LatexOCR (pix2tex) loaded.")
    except Exception as e:
        import traceback
        print(f" [!] Error loading Math OCR: {e}")
        traceback.print_exc()
        print("     Continuing without math OCR capability...")
        _math_model = None

    return True


# ==========================================
# 4. MAIN ANALYZE FUNCTION
# ==========================================
def analyze_vectors(vectors):
    """
    Main entry point. Receives list of list of (x,y,t).
    Returns final LaTeX string.
    """
    if not vectors:
        return ""

    # A. Load models if not loaded
    if _lstm_model is None:
        if not load_models():
            return ""

    # B. Feature Extraction & Normalization
    print(f"\nAnalyzing {len(vectors)} strokes...")
    features = extract_features_from_vectors(vectors)

    # Sanity check: If document is empty, return early
    if len(features) == 0:
        return ""

    # C. Normalization (Load the training scaler)
    try:
        scaler = joblib.load(SCALER_PATH)
        # print("Loaded scaler from training.") # Optional logging
    except FileNotFoundError:
        print(
            f"Error: {SCALER_PATH} not found. You must run the training script once to generate it."
        )
        return ""
    except (EOFError, Exception) as e:
        print(f"Error: scaler.save is corrupted or unreadable: {e}")
        return ""

    # CRITICAL FIX: Transform the features using the loaded scaler.
    # Do NOT use fit_transform (that would erase the "global" size knowledge).
    features_norm = scaler.transform(features)

    # D. Classification (LSTM)
    # Convert to Tensor (Explicitly Float32 to match LSTM weights)
    x_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(
        0
    ).to(_device)  # [1, Seq_Len, Features]
    lengths = torch.tensor([len(features_norm)])

    with torch.no_grad():
        logits = _lstm_model(x_tensor, lengths)
        preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

    # D. Smoothing
    smoothed_preds = smooth_predictions(preds, window_size=5)
    print(f"Predictions: {smoothed_preds}")

    # Validation: Check predictions and vectors match
    if len(smoothed_preds) == 0:
        print("Error: No predictions generated")
        return ""

    if len(vectors) != len(smoothed_preds):
        print(f"Warning: vectors ({len(vectors)}) and predictions ({len(smoothed_preds)}) length mismatch")
        # Use minimum length to avoid index errors
        min_len = min(len(vectors), len(smoothed_preds))
        vectors = vectors[:min_len]
        smoothed_preds = smoothed_preds[:min_len]

    # E. Grouping & OCR Routing
    full_latex = ""

    current_label = smoothed_preds[0]
    current_strokes = []

    # Helper to flush a group
    def process_group(label_idx, stroke_group):
        if not stroke_group:
            return ""

        label_name = LABEL_MAP.get(label_idx, "text")

        # Special handling for math: split by vertical gaps to handle multiple expressions
        if label_name == "math":
            # Check if math model is available
            if _math_model is None:
                print("   -> Math OCR not available (model failed to load)")
                return "\n[Math OCR unavailable]\n"

            # Split math strokes into separate expression groups
            math_subgroups = split_math_groups_by_vertical_gap(stroke_group, gap_threshold=50)
            print(f"   -> Split {len(stroke_group)} math strokes into {len(math_subgroups)} expression(s)")

            result = ""
            for i, subgroup in enumerate(math_subgroups):
                # Render each math expression separately
                img = render_strokes_to_image(subgroup, label_type="math")
                if img is None:
                    continue

                print(f"   -> Processing Math expression {i+1}/{len(math_subgroups)}...")
                try:
                    # Ensure image is in the right format
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Call the appropriate model
                    if USING_PIX2TEXT:
                        # P2TLatexOCR returns a dict with 'text' key
                        result_obj = _math_model(img)
                        if isinstance(result_obj, dict):
                            latex = result_obj.get('text', '').strip()
                        else:
                            latex = str(result_obj).strip()
                        # Remove $$ delimiters if present
                        if latex.startswith('$$') and latex.endswith('$$'):
                            latex = latex[2:-2].strip()
                    else:
                        # pix2tex/LatexOCR returns LaTeX string directly
                        try:
                            latex = _math_model(img, temperature=0.0)
                        except TypeError:
                            latex = _math_model(img)

                    if latex is None or latex.strip() == "":
                        print(f"   -> Math model returned empty result for expression {i+1}")
                        continue

                    print(f"   -> Math OCR result: {latex}")
                    result += f"\n$${latex}$$\n"
                except Exception as e:
                    print(f"Math Error on expression {i+1}: {e}")

            return result

        # For text and diagrams, process normally
        if label_name == "text":
            # Check if text models are available
            if _text_model is None or _text_processor is None:
                print("   -> Text OCR not available (models failed to load)")
                return "[Text OCR unavailable] "

            # Split text strokes into separate lines (TrOCR is a single-line model)
            text_lines = split_math_groups_by_vertical_gap(stroke_group, gap_threshold=50)
            print(f"   -> Split {len(stroke_group)} text strokes into {len(text_lines)} line(s)")

            result = ""
            for i, line_strokes in enumerate(text_lines):
                img = render_strokes_to_image(line_strokes, label_type="text")
                if img is None:
                    continue

                # --- SANITIZATION STEP ---
                if isinstance(img, np.ndarray):
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    img = Image.fromarray(img)

                if img.mode != "RGB":
                    img = img.convert("RGB")
                # -------------------------

                print(f"   -> Processing Text line {i+1}/{len(text_lines)}...")
                try:
                    pixel_values = _text_processor(
                        images=img, return_tensors="pt"
                    ).pixel_values.to(_device)

                    _text_model.eval()

                    generated_ids = _text_model.generate(
                        pixel_values,
                        num_beams=5,
                        repetition_penalty=1.5, # Reduce repeated characters
                    )
                    text = _text_processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()

                    # TrOCR often hallucinates a trailing " ." — strip it
                    if text.endswith(" ."):
                        text = text[:-2].strip()

                    # Detect list formatting from OCR output
                    text = _format_list_item(text)
                    result += f"{text}\n"
                except Exception as e:
                    print(f"   -> Text OCR error on line {i+1}: {e}")
                    result += "[Text OCR error]\n"

            return result

        img = render_strokes_to_image(stroke_group, label_type=label_name)
        if img is None:
            return ""

        result = ""

        if label_name == "diagram":
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

    return full_latex
