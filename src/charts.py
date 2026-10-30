"""Render only the 20 input closes, with fixed axes and no annotations."""
import numpy as np
from PIL import Image, ImageDraw

SIZE = 224
Y_MIN, Y_MAX = 0.80, 1.20


def normalized_prices(closes):
    values = np.asarray(closes, dtype=float)
    if values.shape != (20,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("chart requires exactly 20 finite positive Close values")
    return values / values[0]


def render_chart(closes):
    """Map days 0–19 across x=0–223; y=0.80–1.20 maps to bottom–top.

    Pillow clips line segments at the image boundary. No y-axis autoscaling,
    future price, date, label, or annotation enters the image.
    """
    values = normalized_prices(closes)
    points = [(round(i * (SIZE - 1) / 19),
               round((Y_MAX - float(value)) * (SIZE - 1) / (Y_MAX - Y_MIN)))
              for i, value in enumerate(values)]
    image = Image.new("RGB", (SIZE, SIZE), "white")
    ImageDraw.Draw(image).line(points, fill="black", width=1)
    return image
