from math import isfinite, exp
from time import perf_counter

import cv2
import numpy as np


def read_size(prompt="\nВведите нечётный размер ядра: "):
    while True:
        try:
            size = int(input(prompt).strip())
            if size <= 0 or size % 2 == 0:
                raise ValueError
            return size
        except ValueError:
            print("Размер ядра должен быть положительным нечётным целым числом!")


def _read_sigma(prompt):
    while True:
        try:
            sigma = float(input(prompt).strip().replace(",", "."))
            if sigma <= 0 or not isfinite(sigma):
                raise ValueError
            return sigma
        except ValueError:
            print("Введите положительное число для сигмы!")


def read_sigma_x(prompt="\nВведите sigma_x: "):
    return _read_sigma(prompt)


def read_sigma_y(prompt="\nВведите sigma_y: "):
    return _read_sigma(prompt)


def make_gaussian_kernel(size: int, sigma_x: float, sigma_y: float) -> list[list[float]]:
    kernel = []
    radius = size // 2

    for y in range(-radius, radius + 1):
        row = []
        for x in range(-radius, radius + 1):
            weight = exp(
                -(x * x / (2 * sigma_x * sigma_x)
                  + y * y / (2 * sigma_y * sigma_y))
            )
            row.append(weight)
        kernel.append(row)

    total = sum(sum(row) for row in kernel)

    for y in range(size):
        for x in range(size):
            kernel[y][x] /= total

    return kernel


def blur(image, kernel) -> list[list[float]]:
    size = len(kernel)
    radius = size // 2
    height, width = len(image), len(image[0])
    result = [[0.0 for _ in range(width)] for _ in range(height)]

    for row in range(height):
        for col in range(width):
            value = 0.0

            for ky in range(size):
                for kx in range(size):
                    source_row = min(max(row + ky - radius, 0), height - 1)
                    source_col = min(max(col + kx - radius, 0), width - 1)
                    value += (image[source_row][source_col] * kernel[ky][kx])

            result[row][col] = value

    return result


def blur_opencv(image, size, sigma_x, sigma_y):
    return cv2.GaussianBlur(image, (size, size), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REPLICATE)


def main():
    path = input("Путь к изображению: ").strip().strip('"')
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Не удалось открыть изображение. Проверь путь.")
        return

    size = read_size()
    sigma_x = read_sigma_x()
    sigma_y = read_sigma_y()
    kernel = make_gaussian_kernel(size, sigma_x, sigma_y)

    image_float = image.astype(np.float64)

    start = perf_counter()
    manual_result = blur(image_float, kernel)
    manual_time = perf_counter() - start

    start = perf_counter()
    opencv_result = blur_opencv(image_float, size, sigma_x, sigma_y)
    opencv_time = perf_counter() - start

    manual_result = np.asarray(manual_result)

    difference = np.max(np.abs(manual_result - opencv_result))
    print(f"Максимальная разница между результатами: {difference:.8f}")
    print(f"Ручной метод: {manual_time:.4f} с")
    print(f"OpenCV: {opencv_time:.6f} с")

    manual_to_save = np.clip(np.rint(manual_result), 0, 255).astype(np.uint8)
    opencv_to_save = np.clip(np.rint(opencv_result), 0, 255).astype(np.uint8)

    cv2.imwrite("result_manual.png", manual_to_save)
    cv2.imwrite("result_opencv.png", opencv_to_save)
    print("Результаты сохранены в result_manual.png и result_opencv.png")


if __name__ == "__main__":
    main()
