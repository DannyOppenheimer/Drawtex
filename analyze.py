import os
from PIL import Image, ImageDraw
from gradio_client import Client, handle_file


def analyze_vectors(vectors):
    if not vectors:
        return

    all_x = [point[0] for stroke in vectors for point in stroke]
    all_y = [point[1] for stroke in vectors for point in stroke]

    if not all_x or not all_y:
        return

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    padding = 20
    width = int(max_x - min_x) + (padding * 2)
    height = int(max_y - min_y) + (padding * 2)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for stroke in vectors:
        points = []
        for point in stroke:
            x = point[0] - min_x + padding
            y = point[1] - min_y + padding
            points.append((x, y))

        if len(points) > 1:
            draw.line(points, fill="black", width=3)

        for p in points:
            r = 1.5
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill="black")

    temp_filename = "temp_stroke_input.png"
    image.save(temp_filename)

    try:
        client = Client("oppenheimerd/Drawtex")

        result = client.predict(image=handle_file(temp_filename))

        print(result)
        return result

    except Exception as e:
        print(e)
        return None
