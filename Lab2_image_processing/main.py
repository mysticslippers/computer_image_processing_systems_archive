def open_video(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        cap.release()
        raise OSError(f"Не удалось открыть видео: {path}")

    success, first_frame = cap.read()

    if not success or first_frame is None:
        cap.release()
        raise RuntimeError("Не удалось прочитать первый кадр")

    return cap, first_frame


def prepare_reference(first_frame, roi=None, full_frame=False):
    frame_height, frame_width = first_frame.shape[:2]

    if roi is not None and full_frame:
        raise ValueError("Нельзя одновременно выбрать область и весь кадр")

    if full_frame:
        x, y, width, height = 0, 0, frame_width, frame_height
    elif roi is not None:
        x, y, width, height = roi
    else:
        window_name = "Select object: Enter to confirm, C to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            x, y, width, height = cv2.selectROI(
                window_name,
                first_frame,
                showCrosshair=True,
                fromCenter=False
            )
        finally:
            cv2.destroyWindow(window_name)

    if width <= 0 or height <= 0:
        raise ValueError("Область объекта не выбрана")

    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("Область объекта выходит за границы первого кадра")

    reference_image = first_frame[
        y:y + height,
        x:x + width
    ].copy()

    reference_corners = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ]).reshape(-1, 1, 2)

    return reference_image, reference_corners
