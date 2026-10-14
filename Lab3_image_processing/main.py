import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0

    return array


def load_mask(path: str, image_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as mask_image:
        mask_image = mask_image.convert("L")
        mask = np.asarray(mask_image, dtype=np.uint8)

    if mask.shape != image_shape:
        raise ValueError(f"Размер маски {mask.shape} не совпадает с размером изображения {image_shape}")

    return (mask >= 128).astype(np.float32)
    
