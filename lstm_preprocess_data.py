import json
import os
from PIL import Image, ImageDraw
import numpy as np

# Configuration
INPUT_FILE = "your_data.json"  # Your collected data
OUTPUT_DIR = "dataset"
IMG_SIZE = 224  # Standard size for Hugging Face models
PADDING = 20


def render_stroke_group(strokes, filename):
    """Renders a list of strokes into a single image."""
    # 1. Flatten all points to find bounding box
    all_points = []
    for stroke in strokes:
        # Each point is [x, y, t], we just need x and y
        all_points.extend([p[:2] for p in stroke["points"]])

    if not all_points:
        return

    # 2. Normalize coordinates
    arr = np.array(all_points)
    min_x, min_y = arr.min(axis=0)
    max_x, max_y = arr.max(axis=0)

    width = max_x - min_x
    height = max_y - min_y

    # 3. Create blank white canvas
    # We maintain aspect ratio but fit it into a square for the model
    canvas_dim = int(max(width, height) + PADDING * 2)
    image = Image.new("RGB", (canvas_dim, canvas_dim), "white")
    draw = ImageDraw.Draw(image)

    # 4. Draw strokes centered
    offset_x = (canvas_dim - width) / 2 - min_x
    offset_y = (canvas_dim - height) / 2 - min_y

    for stroke in strokes:
        points = [(p[0] + offset_x, p[1] + offset_y) for p in stroke["points"]]
        # Draw line with width=3 for visibility
        draw.line(points, fill="black", width=3)

    # 5. Resize to standard model input (224x224)
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    image.save(filename)


def process_json(data):
    # Setup directories
    for label in ["text", "math", "diagram"]:
        os.makedirs(f"{OUTPUT_DIR}/{label}", exist_ok=True)

    img_count = 0

    # Iterate through sessions
    for session in data:
        current_group = []
        current_label = None

        for stroke in session["strokes"]:
            label = stroke["label"]

            # If label changes, save the previous group and start new
            if label != current_label and current_group:
                filename = f"{OUTPUT_DIR}/{current_label}/sample_{img_count}.png"
                render_stroke_group(current_group, filename)
                img_count += 1
                current_group = []

            current_label = label
            current_group.append(stroke)

        # Save the last group in the session
        if current_group:
            filename = f"{OUTPUT_DIR}/{current_label}/sample_{img_count}.png"
            render_stroke_group(current_group, filename)
            img_count += 1


# Run
with open(INPUT_FILE, "r") as f:
    raw_data = json.load(f)
process_json(raw_data)
print("Preprocessing complete! Check the 'dataset' folder.")
