import json
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")

    plt.title("Learning Curve (Random Forest)")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def get_stroke_features(current_stroke, all_strokes_in_scene, prev_stroke=None):
    raw_points = current_stroke["points"]

    if len(raw_points) < 2:
        return [0] * 9  # Updated to 9 features

    pts = np.array([p[:2] for p in raw_points])
    timestamps = [p[2] for p in raw_points]

    # --- Geometry ---
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]

    # Pre-calculate current bounding box for containment checks
    # Format: [min_x, min_y, max_x, max_y]
    curr_bbox = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]

    diffs = np.diff(pts, axis=0)
    path_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
    displacement = np.linalg.norm(pts[-1] - pts[0])
    linearity = displacement / (path_length + 1e-5)

    duration = timestamps[-1] - timestamps[0]
    speed = path_length / (duration + 1e-5)

    # --- Context 1: Time/Distance Gap ---
    time_gap = 0
    dist_gap = 0
    if prev_stroke:
        prev_points = prev_stroke["points"]
        prev_end_time = prev_points[-1][2]
        prev_end_pos = np.array(prev_points[-1][:2])
        curr_start_pos = np.array(raw_points[0][:2])

        raw_time_gap = timestamps[0] - prev_end_time
        time_gap = min(raw_time_gap, 5.0)
        dist_gap = np.linalg.norm(curr_start_pos - prev_end_pos)

    # --- Context 2: Density & Containment (The New Stuff) ---
    curr_center = np.mean(pts, axis=0)
    density_count = 0
    enclosed_by_count = 0
    encloses_count = 0

    density_radius = 50.0

    for other in all_strokes_in_scene:
        if other is current_stroke:
            continue

        # Get other stroke's geometry
        o_pts = np.array([p[:2] for p in other["points"]])
        o_center = np.mean(o_pts, axis=0)

        # 1. Density Check (Proximity)
        dist = np.linalg.norm(curr_center - o_center)
        if dist < density_radius:
            density_count += 1

        # 2. Containment Check
        # Does 'other' surround 'current'?
        o_min = np.min(o_pts, axis=0)
        o_max = np.max(o_pts, axis=0)
        o_bbox = [o_min[0], o_min[1], o_max[0], o_max[1]]

        # Check: Is Current inside Other?
        if (
            curr_bbox[0] >= o_bbox[0]
            and curr_bbox[1] >= o_bbox[1]
            and curr_bbox[2] <= o_bbox[2]
            and curr_bbox[3] <= o_bbox[3]
        ):
            enclosed_by_count += 1

        # Check: Is Other inside Current?
        elif (
            o_bbox[0] >= curr_bbox[0]
            and o_bbox[1] >= curr_bbox[1]
            and o_bbox[2] <= curr_bbox[2]
            and o_bbox[3] <= curr_bbox[3]
        ):
            encloses_count += 1

    # --- Feature 10: Vertical Baseline Deviation (Targeting Math) ---
    vertical_deviation = 0

    if prev_stroke:
        prev_points = prev_stroke["points"]
        prev_pts_array = np.array([p[:2] for p in prev_points])

        # Calculate Y-centers (Vertical Centers)
        curr_y_center = np.mean(pts[:, 1])
        prev_y_center = np.mean(prev_pts_array[:, 1])

        # Calculate Height of previous stroke to normalize
        # (A 10px jump matters more for small text than a giant diagram)
        prev_h = (np.max(prev_pts_array[:, 1]) - np.min(prev_pts_array[:, 1])) + 1e-5

        # How much did we jump up/down relative to the previous stroke's size?
        # Standard Text ≈ 0.0 to 0.1
        # Math (superscripts/fractions) ≈ 0.5 to 1.0+
        vertical_deviation = abs(curr_y_center - prev_y_center) / prev_h

    # --- Feature 11: Tortuosity (Path Efficiency) ---
    # Calculate the diagonal length of the bounding box
    bbox_diagonal = np.sqrt(width**2 + height**2) + 1e-5

    # Ratio: How much ink did we use vs. how much space did we take up?
    # High = Squiggly/Dense (Math). Low = Efficient (Diagram/Simple Text)
    tortuosity = path_length / bbox_diagonal

    total_turn_angle = 0.0
    
    if len(pts) > 2:
        # 1. Calculate vectors for every segment
        # v1 is vector from p0->p1, p1->p2, etc.
        # v2 is vector from p1->p2, p2->p3, etc.
        diffs = np.diff(pts, axis=0)
        
        # 2. Calculate angles for every vector (in radians)
        # arctan2 handles the quadrants correctly
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        # 3. Calculate the difference between consecutive angles
        # This gives us the "turn" at every point
        angle_changes = np.diff(angles)
        
        # 4. Handle "Wrap Around" (e.g. 359 degrees -> 1 degree is a small turn, not huge)
        # If change is > PI, subtract 2PI. If < -PI, add 2PI.
        angle_changes = np.mod(angle_changes + np.pi, 2 * np.pi) - np.pi
        
        # 5. Sum absolute values
        total_turn_angle = np.sum(np.abs(angle_changes))

    horiz_overlap_ratio = 0.0
    
    if prev_stroke:
        # Get X-range of previous stroke
        p_min_x = np.min(np.array([p[0] for p in prev_stroke["points"]]))
        p_max_x = np.max(np.array([p[0] for p in prev_stroke["points"]]))
        
        # Get X-range of current stroke
        c_min_x = min_xy[0]
        c_max_x = max_xy[0]
        
        # Calculate intersection of the X-intervals
        overlap_start = max(p_min_x, c_min_x)
        overlap_end = min(p_max_x, c_max_x)
        overlap_len = max(0, overlap_end - overlap_start)
        
        # Normalize by the width of the smaller stroke 
        # (Avoids bias when a small "2" is above a giant fraction bar)
        min_width = min(c_max_x - c_min_x, p_max_x - p_min_x) + 1e-5
        
        horiz_overlap_ratio = overlap_len / min_width

    return [
        width,
        height,
        width / (height + 1e-5),
        linearity,
        speed,
        time_gap,
        dist_gap,
        density_count,
        enclosed_by_count,
        vertical_deviation,
        tortuosity,
        total_turn_angle,
        horiz_overlap_ratio
    ]


def prepare_dataset(json_file):
    with open(json_file, "r") as f:
        sessions = json.load(f)

    X = []
    y = []

    print(f"Loading data from {json_file}...")

    for session in sessions:
        if "strokes" not in session:
            continue

        strokes_list = session["strokes"]
        if not strokes_list:
            continue

        strokes_list.sort(key=lambda s: s["points"][0][2])

        prev_stroke = None

        for stroke in strokes_list:
            label = stroke.get("label")

            if label:
                features = get_stroke_features(stroke, strokes_list, prev_stroke)
                X.append(features)
                y.append(label)

            prev_stroke = stroke

    return np.array(X), np.array(y)


def smooth_predictions(predictions, window_size=3):
    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - 1)
        end = min(len(predictions), i + 2)
        window = predictions[start:end]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed


def get_bbox(strokes):
    all_x = []
    all_y = []
    for s in strokes:
        for p in s["points"]:
            all_x.append(p[0])
            all_y.append(p[1])

    if not all_x:
        return (0, 0, 0, 0)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def rect_intersects(r1, r2):
    return not (
        r2[0] > r1[0] + r1[2]
        or r2[0] + r2[2] < r1[0]
        or r2[1] > r1[1] + r1[3]
        or r2[1] + r2[3] < r1[1]
    )


def segment_and_merge(strokes, labels):
    segments = []
    if not strokes:
        return []

    current_segment = {"type": labels[0], "strokes": [strokes[0]]}

    for i in range(1, len(strokes)):
        if labels[i] == current_segment["type"]:
            current_segment["strokes"].append(strokes[i])
        else:
            segments.append(current_segment)
            current_segment = {"type": labels[i], "strokes": [strokes[i]]}
    segments.append(current_segment)

    merged_segments = []
    i = 0
    while i < len(segments):
        curr = segments[i]

        if curr["type"] == "text":
            if merged_segments and merged_segments[-1]["type"] == "diagram":
                prev_diag = merged_segments[-1]

                text_rect = get_bbox(curr["strokes"])
                diag_rect = get_bbox(prev_diag["strokes"])

                padding = 40
                inflated_diag = (
                    diag_rect[0] - padding,
                    diag_rect[1] - padding,
                    diag_rect[2] + padding * 2,
                    diag_rect[3] + padding * 2,
                )

                if rect_intersects(inflated_diag, text_rect):
                    print(
                        f"   -> Merging label '{curr['type']}' into previous Diagram."
                    )
                    prev_diag["strokes"].extend(curr["strokes"])
                    i += 1
                    continue

        merged_segments.append(curr)
        i += 1

    return merged_segments


if __name__ == "__main__":
    try:
        X, y = prepare_dataset("collected_strokes.json")
    except FileNotFoundError:
        print("Error: 'collected_strokes.json' not found.")
        exit()
    except json.JSONDecodeError:
        print("Error: Your JSON file is malformed.")
        exit()

    if len(X) == 0:
        print("Error: No valid strokes found in JSON.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} strokes, validating on {len(X_test)}...")

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=15,  # NEW: Prevents infinite complexity
        min_samples_leaf=4,  # NEW: Requires at least 4 strokes to create a rule
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Model Accuracy ---")
    print(classification_report(y_test, y_pred))

    with open("collected_strokes.json", "r") as f:
        full_data = json.load(f)

        test_scene = []
        for session in full_data:
            if session.get("strokes"):
                test_scene = session["strokes"]
                test_scene.sort(key=lambda s: s["points"][0][2])
                break

    if test_scene:
        print("\n--- Running Segmentation Pipeline on Test Scene ---")

        scene_features = []
        prev_s = None
        for s in test_scene:
            scene_features.append(get_stroke_features(s, test_scene, prev_s))
            prev_s = s

        raw_predictions = clf.predict(scene_features)
        smoothed_labels = smooth_predictions(raw_predictions)
        final_segments = segment_and_merge(test_scene, smoothed_labels)

        print(f"Found {len(final_segments)} distinct groups:")
        for idx, seg in enumerate(final_segments):
            n_strokes = len(seg["strokes"])
            print(f"  Group {idx+1}: Type='{seg['type']}', Strokes={n_strokes}")

    # Get the importance scores
    importances = clf.feature_importances_
    feature_names = [
        "Width",
        "Height",
        "Aspect Ratio",
        "Linearity",
        "Speed",
        "Time Gap",
        "Dist Gap",
        "Density",
        "Enclosed By",
        "Vertical Baseline Deviation",
        "Tortuosity",
        "Total Turn Angle",
        "Horizontal Overlap"
    ]

    # Sort them to see the winners
    indices = np.argsort(importances)[::-1]

    print("\n--- Feature Rankings ---")
    for f in range(len(feature_names)):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

    plot_learning_curve(clf, X, y)
