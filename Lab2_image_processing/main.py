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
