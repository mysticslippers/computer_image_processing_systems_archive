from pathlib import Path

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


def create_test_masks(image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    height, width = image_shape

    if height < 2 or width < 2:
        raise ValueError("Размер изображения должен быть хотя бы 2×2")

    zeros_mask = np.zeros(image_shape, dtype=np.float32)
    ones_mask = np.ones(image_shape, dtype=np.float32)

    mixed_mask = np.zeros(image_shape, dtype=np.float32)
    mixed_mask[
        height // 4:3 * height // 4,
        width // 4:3 * width // 4
    ] = 1.0

    return {
        "zeros": zeros_mask,
        "ones": ones_mask,
        "mixed": mixed_mask,
    }


def save_image(image: np.ndarray, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.clip(image, 0.0, 1.0)
    array = np.round(array * 255.0).astype(np.uint8)

    Image.fromarray(array).save(output_path)
