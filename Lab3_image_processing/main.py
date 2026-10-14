import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0

    return array
