"""Tests for vision module: object detection and tracking."""

import numpy as np
import pytest

from src.vision.detection import Detection, ObjectDetector
from src.vision.tracking import ObjectTracker, TrackedObject


# ─────────────────────────────────────────────────────────────
# Detection dataclass
# ─────────────────────────────────────────────────────────────

class TestDetection:
    def test_detection_fields(self):
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(10, 20, 50, 80),
            center=(30, 50),
            area=2400,
        )
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == pytest.approx(0.9)
        assert det.bbox == (10, 20, 50, 80)
        assert det.center == (30, 50)
        assert det.area == 2400


# ─────────────────────────────────────────────────────────────
# ObjectDetector
# ─────────────────────────────────────────────────────────────

class TestObjectDetector:
    def test_init_defaults(self):
        detector = ObjectDetector()
        assert detector.confidence_threshold == pytest.approx(0.5)
        assert detector.device == "cpu"
        assert not detector._is_initialized

    def test_initialize_without_yolo(self):
        detector = ObjectDetector(model_path="nonexistent.pt")
        result = detector.initialize()
        assert result is True
        assert detector._is_initialized

    def test_detect_returns_list(self):
        detector = ObjectDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)

    def test_mock_detect_returns_detection(self):
        detector = ObjectDetector()
        detector._is_initialized = True  # skip real init
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # model is None → falls back to _mock_detect
        detections = detector.detect(frame)
        assert len(detections) == 1
        assert detections[0].class_name == "person"
        assert detections[0].confidence == pytest.approx(0.85)

    def test_mock_detect_center_within_frame(self):
        detector = ObjectDetector()
        detector._is_initialized = True
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        h, w = frame.shape[:2]
        cx, cy = detections[0].center
        assert 0 <= cx < w
        assert 0 <= cy < h

    def test_detect_obstacles_filters_to_obstacle_classes(self):
        detector = ObjectDetector()
        detector._is_initialized = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # mock returns class_id=0 ("person") which IS in OBSTACLE_CLASSES
        obstacles = detector.detect_obstacles(frame)
        assert all(d.class_id in ObjectDetector.OBSTACLE_CLASSES for d in obstacles)

    def test_get_obstacle_map_shape(self):
        detector = ObjectDetector()
        detector._is_initialized = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        grid = detector.get_obstacle_map(frame, grid_size=(10, 10))
        assert grid.shape == (10, 10)
        assert grid.dtype == np.uint8

    def test_get_obstacle_map_marks_obstacle(self):
        detector = ObjectDetector()
        detector._is_initialized = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        grid = detector.get_obstacle_map(frame, grid_size=(10, 10))
        # At least one cell should be occupied (mock detect returns a person)
        assert grid.sum() >= 1

    def test_obstacle_classes_dict_not_empty(self):
        assert len(ObjectDetector.OBSTACLE_CLASSES) > 0


# ─────────────────────────────────────────────────────────────
# ObjectTracker
# ─────────────────────────────────────────────────────────────

class TestObjectTracker:
    def test_init_defaults(self):
        tracker = ObjectTracker()
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == pytest.approx(0.3)
        assert tracker.next_id == 0
        assert len(tracker.tracks) == 0

    def test_update_empty_detections(self):
        tracker = ObjectTracker()
        result = tracker.update([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_update_creates_tracks(self):
        tracker = ObjectTracker(min_hits=1)
        det = (100, 100, 200, 200, "car")
        tracker.update([det])
        assert len(tracker.tracks) == 1

    def test_track_confirmed_after_min_hits(self):
        tracker = ObjectTracker(min_hits=3)
        det = (100, 100, 200, 200, "car")
        for _ in range(3):
            confirmed = tracker.update([det])
        # After 3 updates the track should be confirmed
        assert len(confirmed) == 1

    def test_track_id_increments(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([(0, 0, 50, 50, "person")])
        tracker.update([(200, 200, 250, 250, "car")])  # no overlap → new track
        assert tracker.next_id == 2

    def test_stale_tracks_removed(self):
        tracker = ObjectTracker(max_age=2, min_hits=1)
        tracker.update([(0, 0, 50, 50, "person")])
        # Send empty updates to age out the track
        tracker.update([])
        tracker.update([])
        tracker.update([])  # exceeds max_age=2
        assert len(tracker.tracks) == 0

    def test_iou_matching_updates_existing_track(self):
        tracker = ObjectTracker(min_hits=1, iou_threshold=0.1)
        det = (100, 100, 200, 200, "car")
        tracker.update([det])
        initial_id = tracker.next_id
        # Slightly moved detection should match existing track
        tracker.update([(105, 105, 205, 205, "car")])
        assert tracker.next_id == initial_id  # no new track created

    def test_predict_positions_returns_dict(self):
        tracker = ObjectTracker(min_hits=1)
        det = (100, 100, 200, 200, "car")
        for _ in range(3):
            tracker.update([det])
        preds = tracker.predict_positions(frames_ahead=5)
        assert isinstance(preds, dict)

    def test_compute_iou_identical_boxes(self):
        tracker = ObjectTracker()
        box = (0, 0, 100, 100)
        assert tracker._compute_iou(box, box) == pytest.approx(1.0)

    def test_compute_iou_non_overlapping(self):
        tracker = ObjectTracker()
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert tracker._compute_iou(box1, box2) == pytest.approx(0.0)

    def test_compute_iou_partial_overlap(self):
        tracker = ObjectTracker()
        box1 = (0, 0, 10, 10)
        box2 = (5, 0, 15, 10)
        iou = tracker._compute_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_velocity_estimated_after_update(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([(100, 100, 200, 200, "car")])
        tracker.update([(110, 110, 210, 210, "car")])
        track = list(tracker.tracks.values())[0]
        assert track.velocity == (10, 10)

    def test_history_trimmed_to_30(self):
        tracker = ObjectTracker(min_hits=1)
        det_base = (100, 100, 200, 200, "car")
        tracker.update([det_base])
        for i in range(40):
            tracker.update([(100 + i, 100 + i, 200 + i, 200 + i, "car")])
        track = list(tracker.tracks.values())[0]
        assert len(track.history) <= 30
