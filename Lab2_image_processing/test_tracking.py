import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from main import (
    draw_tracking,
    extract_features,
    locate_object,
    match_features,
    open_video,
    prepare_reference,
    process_video,
)


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        texture = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.6)
        cls.reference = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        cv2.putText(cls.reference, "TRACK", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)
        cv2.circle(cls.reference, (230, 170), 22, (0, 0, 0), 3)
        cls.transform = np.float64([
            [0.62, -0.08, 70],
            [0.06, 0.62, 25],
            [0.00015, 0.0001, 1],
        ])
        cls.moved = cv2.warpPerspective(cls.reference, cls.transform, (320, 240))

    def test_estimated_corners_match_known_perspective_transform(self):
        reference, corners = prepare_reference(self.reference, full_frame=True)
        detector = cv2.SIFT_create(nfeatures=2000)
        reference_keypoints, reference_descriptors = extract_features(reference, detector)
        frame_keypoints, frame_descriptors = extract_features(self.moved, detector)
        matches = match_features(reference_descriptors, frame_descriptors, cv2.BFMatcher(cv2.NORM_L2))
        actual = locate_object(reference_keypoints, frame_keypoints, matches, corners)
        self.assertIsNotNone(actual)
        expected = cv2.perspectiveTransform(corners, self.transform)
        max_error = np.linalg.norm(actual.reshape(4, 2) - expected.reshape(4, 2), axis=1).max()
        self.assertLess(max_error, 3.0)

    def test_video_keeps_first_frame_and_recovers_after_object_disappears(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source = Path(directory) / "source.avi"
            output = Path(directory) / "tracked.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12, (320, 240))
            self.assertTrue(writer.isOpened())
            try:
                for frame in (self.reference, self.moved, np.zeros_like(self.reference), self.moved):
                    writer.write(frame)
            finally:
                writer.release()

            stats = process_video(source, output, full_frame=True, display=False)
            self.assertEqual(stats["frames_processed"], 4)
            self.assertEqual(stats["frames_detected"], 3)
            self.assertFalse(stats["stopped_by_user"])
            self.assertTrue(output.with_suffix(".json").is_file())

            cap = cv2.VideoCapture(str(output))
            try:
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 12, places=2)
                decoded = []
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    self.assertEqual(frame.shape, (240, 320, 3))
                    decoded.append(frame)
            finally:
                cap.release()
            self.assertEqual(len(decoded), 4)
            # Чёрный кадр должен остаться без ложной рамки.
            self.assertLess(float(decoded[2].mean()), 2)
            # После пропажи объект снова найден и рамка появилась.
            green = decoded[3][:, :, 1].astype(np.int16)
            red = decoded[3][:, :, 2].astype(np.int16)
            self.assertGreater(int(np.count_nonzero(green - red > 70)), 100)

    def test_invalid_roi_is_rejected(self):
        for roi in ((-1, 0, 20, 20), (0, 0, 0, 10), (310, 230, 30, 30)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                prepare_reference(self.reference, roi=roi)

    def test_missing_video_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with self.assertRaises(OSError):
                open_video(Path(directory) / "missing.avi")

    def test_headless_run_requires_explicit_reference(self):
        with self.assertRaises(ValueError):
            process_video("unused.avi", display=False)

    def test_input_video_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            process_video("same.avi", "same.avi", full_frame=True, display=False)

    def test_annotation_does_not_modify_source_frame(self):
        _, corners = prepare_reference(self.reference, full_frame=True)
        original = self.reference.copy()
        annotated = draw_tracking(self.reference, corners, "Object")
        np.testing.assert_array_equal(self.reference, original)
        self.assertFalse(np.array_equal(annotated, original))


if __name__ == "__main__":
    unittest.main()
