import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


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


def extract_features(image, detector):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = detector.detectAndCompute(
        gray_image,
        None
    )

    return keypoints, descriptors


def match_features(reference_descriptors, frame_descriptors, matcher, ratio=0.75):
    if reference_descriptors is None or frame_descriptors is None:
        return []

    if len(frame_descriptors) < 2:
        return []

    matches = matcher.knnMatch(
        reference_descriptors,
        frame_descriptors,
        k=2
    )

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        best, second = pair

        if best.distance < ratio * second.distance:
            good_matches.append(best)

    return good_matches


def locate_object(reference_keypoints, frame_keypoints, matches, reference_corners, min_matches=10, min_inliers=8):
    if len(matches) < max(4, min_matches):
        return None

    reference_points = np.float32([
        reference_keypoints[match.queryIdx].pt
        for match in matches
    ]).reshape(-1, 1, 2)

    frame_points = np.float32([
        frame_keypoints[match.trainIdx].pt
        for match in matches
    ]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        reference_points,
        frame_points,
        cv2.RANSAC,
        5.0
    )

    if homography is None or inlier_mask is None:
        return None

    if inlier_mask.sum() < min_inliers:
        return None

    inlier_points = reference_points[inlier_mask.ravel().astype(bool)]
    covered_area = cv2.contourArea(cv2.convexHull(inlier_points))
    if covered_area < 0.05 * cv2.contourArea(reference_corners):
        return None

    return transform_corners(reference_corners, homography)


def transform_corners(reference_corners, homography):
    object_corners = cv2.perspectiveTransform(reference_corners, homography)

    if not np.isfinite(object_corners).all():
        return None

    if not cv2.isContourConvex(object_corners):
        return None

    if cv2.contourArea(object_corners) < 25:
        return None

    return object_corners


def track_keypoints(previous_gray, current_gray, reference_points, previous_points, reference_corners):
    if previous_gray is None or previous_points is None or len(previous_points) < 8:
        return None, None, None

    parameters = {
        "winSize": (31, 31),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    }
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, previous_points, None, **parameters
    )
    if current_points is None:
        return None, None, None

    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, current_points, None, **parameters
    )
    if backward_points is None:
        return None, None, None

    round_trip_error = np.linalg.norm(
        backward_points.reshape(-1, 2) - previous_points.reshape(-1, 2), axis=1
    )
    points = current_points.reshape(-1, 2)
    height, width = current_gray.shape
    good = (
        forward_status.ravel().astype(bool)
        & backward_status.ravel().astype(bool)
        & (round_trip_error < 2.0)
        & np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0) & (points[:, 0] < width)
        & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    if good.sum() < 8:
        return None, None, None

    reference_points = reference_points[good]
    current_points = current_points[good]
    homography, inlier_mask = cv2.findHomography(
        reference_points, current_points, cv2.RANSAC, 3.0
    )
    if homography is None or inlier_mask is None or inlier_mask.sum() < 8:
        return None, None, None

    corners = transform_corners(reference_corners, homography)
    if corners is None:
        return None, None, None

    inliers = inlier_mask.ravel().astype(bool)
    return corners, reference_points[inliers], current_points[inliers]


def update_tracked_points(reference_keypoints, frame_keypoints, matches, reference_corners, object_corners, tracked_reference=None, tracked_frame=None):
    reference_points = np.float32([
        reference_keypoints[match.queryIdx].pt for match in matches
    ]).reshape(-1, 1, 2)
    frame_points = np.float32([
        frame_keypoints[match.trainIdx].pt for match in matches
    ]).reshape(-1, 1, 2)

    if tracked_reference is not None:
        reference_points = np.concatenate((reference_points, tracked_reference))
        frame_points = np.concatenate((frame_points, tracked_frame))

    homography = cv2.getPerspectiveTransform(
        reference_corners.reshape(4, 2), object_corners.reshape(4, 2)
    )
    projected = cv2.perspectiveTransform(reference_points, homography)
    errors = np.linalg.norm(projected.reshape(-1, 2) - frame_points.reshape(-1, 2), axis=1)
    good = errors < 5.0
    reference_points, frame_points = reference_points[good], frame_points[good]
    _, unique = np.unique(reference_points.reshape(-1, 2), axis=0, return_index=True)
    return reference_points[unique], frame_points[unique]


def replenish_keypoints(gray_image, reference_corners, object_corners, reference_points, frame_points):
    if len(frame_points) >= 80:
        return reference_points, frame_points

    mask = np.zeros_like(gray_image)
    cv2.fillConvexPoly(mask, np.rint(object_corners).astype(np.int32), 255)
    for point in frame_points.reshape(-1, 2):
        cv2.circle(mask, tuple(np.rint(point).astype(int)), 3, 0, cv2.FILLED)

    new_points = cv2.goodFeaturesToTrack(
        gray_image, maxCorners=120 - len(frame_points),
        qualityLevel=0.01, minDistance=3, mask=mask, blockSize=3
    )
    if new_points is None:
        return reference_points, frame_points

    inverse_homography = cv2.getPerspectiveTransform(
        object_corners.reshape(4, 2), reference_corners.reshape(4, 2)
    )
    new_reference = cv2.perspectiveTransform(new_points, inverse_homography)
    points = new_reference.reshape(-1, 2)
    reference_width, reference_height = reference_corners.reshape(4, 2).max(axis=0)
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0) & (points[:, 0] <= reference_width)
        & (points[:, 1] >= 0) & (points[:, 1] <= reference_height)
    )
    return (
        np.concatenate((reference_points, new_reference[valid])),
        np.concatenate((frame_points, new_points[valid]))
    )


def draw_tracking(frame, corners, label="Object"):
    result = frame.copy()

    if corners is None:
        return result

    polygon = np.rint(corners).astype(np.int32)
    cv2.polylines(result, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    height, width = frame.shape[:2]
    points = corners.reshape(4, 2)
    text_x = int(np.clip(points[:, 0].min(), 0, max(0, width - text_width - 8)))
    text_y = int(np.clip(points[:, 1].min() - 8, text_height + 8, height - baseline - 4))

    cv2.rectangle(
        result,
        (text_x, text_y - text_height - 4),
        (text_x + text_width + 8, text_y + baseline + 4),
        (0, 0, 0),
        cv2.FILLED
    )
    cv2.putText(
        result, label, (text_x + 4, text_y), font, scale,
        (0, 255, 0), thickness, cv2.LINE_AA
    )

    return result


def process_video(input_path, output_path=None, label="Object", roi=None, full_frame=False, display=True):
    if not display and roi is None and not full_frame:
        raise ValueError("Для запуска без окон укажите --full-frame или --roi")

    input_path = Path(input_path)
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "outputs" / f"{input_path.stem}_tracked.avi"
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Входное и выходное видео должны иметь разные пути")

    codecs = {".avi": "MJPG", ".mp4": "mp4v"}
    if output_path.suffix.lower() not in codecs:
        raise ValueError("Выходное видео должно иметь расширение .avi или .mp4")

    cap, first_frame = open_video(input_path)
    writer = None
    try:
        reference_image, reference_corners = prepare_reference(
            first_frame, roi=roi, full_frame=full_frame
        )
        detector = cv2.SIFT_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        reference_keypoints, reference_descriptors = extract_features(
            reference_image, detector
        )

        if reference_descriptors is None or len(reference_keypoints) < 10:
            raise ValueError("На эталоне мало ключевых точек: выберите объект с более выраженным рисунком")

        height, width = first_frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Не удалось определить частоту кадров исходного видео")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codecs[output_path.suffix.lower()]),
            fps,
            (width, height)
        )
        if not writer.isOpened():
            raise OSError(f"Не удалось создать выходное видео: {output_path}")

        frames_processed = 0
        frames_detected = 0
        frames_sift = 0
        frames_flow = 0
        stopped_by_user = False
        previous_gray = None
        tracked_reference = None
        tracked_frame = None
        frame = first_frame
        started = perf_counter()

        while frame is not None:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow_corners, flow_reference, flow_frame = track_keypoints(
                previous_gray, current_gray, tracked_reference,
                tracked_frame, reference_corners
            )
            frame_keypoints, frame_descriptors = extract_features(frame, detector)
            matches = match_features(reference_descriptors, frame_descriptors, matcher)
            sift_corners = locate_object(
                reference_keypoints, frame_keypoints, matches, reference_corners
            )
            if sift_corners is not None and flow_corners is not None:
                discrepancy = np.linalg.norm(
                    sift_corners.reshape(4, 2) - flow_corners.reshape(4, 2), axis=1
                ).mean()
                allowed = max(3.0, 0.1 * np.sqrt(cv2.contourArea(flow_corners)))
                if discrepancy > allowed:
                    sift_corners = None

            corners = sift_corners if sift_corners is not None else flow_corners

            if corners is not None:
                points = corners.reshape(4, 2)
                outside_frame = (
                    points[:, 0].max() < 0 or points[:, 0].min() >= width
                    or points[:, 1].max() < 0 or points[:, 1].min() >= height
                )
                too_large = (
                    cv2.contourArea(corners) > 4 * width * height
                    or np.abs(points).max() > 8 * max(width, height)
                )
                if outside_frame or too_large:
                    corners = None

            if corners is None:
                tracked_reference, tracked_frame = None, None
            elif sift_corners is not None:
                frames_sift += 1
                tracked_reference, tracked_frame = update_tracked_points(
                    reference_keypoints, frame_keypoints, matches,
                    reference_corners, sift_corners, flow_reference, flow_frame
                )
            else:
                frames_flow += 1
                tracked_reference, tracked_frame = flow_reference, flow_frame

            if corners is not None:
                tracked_reference, tracked_frame = replenish_keypoints(
                    current_gray, reference_corners, corners,
                    tracked_reference, tracked_frame
                )

            previous_gray = current_gray

            annotated_frame = draw_tracking(frame, corners, label)
            writer.write(annotated_frame)
            frames_processed += 1
            frames_detected += int(corners is not None)

            if frames_processed % 100 == 0:
                print(f"{input_path.name}: обработано {frames_processed} кадров", flush=True)

            if display:
                cv2.imshow("Tracking: Q or Esc to stop", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    stopped_by_user = True
                    break

            success, next_frame = cap.read()
            frame = next_frame if success else None

        elapsed = perf_counter() - started
        stats = {
            "input_video": str(input_path.resolve()),
            "output_video": str(output_path.resolve()),
            "label": label,
            "width": width,
            "height": height,
            "source_fps": fps,
            "reference_keypoints": len(reference_keypoints),
            "frames_processed": frames_processed,
            "frames_detected": frames_detected,
            "frames_detected_by_sift": frames_sift,
            "frames_detected_by_optical_flow": frames_flow,
            "detection_rate": frames_detected / frames_processed,
            "processing_seconds": round(elapsed, 3),
            "processing_fps": round(frames_processed / elapsed, 2),
            "stopped_by_user": stopped_by_user
        }
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    output_path.with_suffix(".json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Трекинг плоского объекта по ключевым точкам SIFT"
    )
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "test-videos" / "mona-lisa.avi",
        help="путь к исходному видео"
    )
    parser.add_argument("-o", "--output", type=Path, help="путь к результату (.avi или .mp4)")
    parser.add_argument("--label", default="Object", help="подпись объекта латиницей")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="область эталона")
    selection.add_argument("--full-frame", action="store_true", help="использовать весь первый кадр как эталон")
    parser.add_argument("--no-display", action="store_true", help="сохранить результат без показа видео")
    args = parser.parse_args()

    if args.no_display and args.roi is None and not args.full_frame:
        parser.error("для --no-display необходимо указать --full-frame или --roi")

    try:
        stats = process_video(
            args.input, args.output, args.label, args.roi,
            args.full_frame, display=not args.no_display
        )
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Ошибка: {error}\n")

    print(f"Результат: {stats['output_video']}")
    print(
        f"Объект найден в {stats['frames_detected']} из "
        f"{stats['frames_processed']} кадров ({stats['detection_rate']:.1%})"
    )


if __name__ == "__main__":
    main()
