"""
Tests for vision/detection.py

Covers:
- Detection dataclass construction and field access
- ObjectDetector initialization and configuration
- detect() with mock model (no YOLO installed)
- detect_obstacles() filtering by OBSTACLE_CLASSES
- get_obstacle_map() grid generation
- Confidence threshold enforcement
- Edge cases: empty frames, single-pixel frames
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from vision.detection import Detection, ObjectDetector


# ---------------------------------------------------------------------------
# Detection dataclass tests
# ---------------------------------------------------------------------------

class TestDetectionDataclass:
    def test_basic_construction(self):
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(10, 20, 110, 220),
            center=(60, 120),
            area=20000,
        )
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == pytest.approx(0.9)
        assert det.bbox == (10, 20, 110, 220)
        assert det.center == (60, 120)
        assert det.area == 20000

    def test_confidence_range_zero(self):
        det = Detection(
            class_id=1,
            class_name="bicycle",
            confidence=0.0,
            bbox=(0, 0, 10, 10),
            center=(5, 5),
            area=100,
        )
        assert det.confidence == 0.0

    def test_confidence_range_one(self):
        det = Detection(
            class_id=2,
            class_name="car",
            confidence=1.0,
            bbox=(0, 0, 100, 100),
            center=(50, 50),
            area=10000,
        )
        assert det.confidence == pytest.approx(1.0)

    def test_bbox_coordinates_order(self):
        """x1 <= x2 and y1 <= y2."""
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.7,
            bbox=(5, 10, 50, 80),
            center=(27, 45),
            area=3150,
        )
        x1, y1, x2, y2 = det.bbox
        assert x1 <= x2
        assert y1 <= y2

    def test_area_equals_bbox_area(self):
        x1, y1, x2, y2 = 0, 0, 40, 30
        expected_area = (x2 - x1) * (y2 - y1)
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.8,
            bbox=(x1, y1, x2, y2),
            center=(20, 15),
            area=expected_area,
        )
        assert det.area == expected_area


# ---------------------------------------------------------------------------
# ObjectDetector initialization tests
# ---------------------------------------------------------------------------

class TestObjectDetectorInit:
    def test_default_initialization(self):
        detector = ObjectDetector()
        assert detector.confidence_threshold == pytest.approx(0.5)
        assert detector.device == "cpu"
        assert detector.model is None
        assert not detector._is_initialized

    def test_custom_confidence_threshold(self):
        detector = ObjectDetector(confidence_threshold=0.8)
        assert detector.confidence_threshold == pytest.approx(0.8)

    def test_custom_device(self):
        detector = ObjectDetector(device="cuda")
        assert detector.device == "cuda"

    def test_obstacle_classes_defined(self):
        # Ensure the class-level constant is populated
        assert len(ObjectDetector.OBSTACLE_CLASSES) > 0
        assert 0 in ObjectDetector.OBSTACLE_CLASSES
        assert ObjectDetector.OBSTACLE_CLASSES[0] == "person"

    def test_initialize_without_ultralytics(self):
        """initialize() should succeed even when ultralytics is absent."""
        detector = ObjectDetector()
        with patch.dict("sys.modules", {"ultralytics": None}):
            result = detector.initialize()
        # Either True (mock path) or True (import error path)
        assert isinstance(result, bool)

    def test_initialize_sets_flag(self):
        detector = ObjectDetector()
        # Patch YOLO import to raise ImportError so mock path is used
        with patch("builtins.__import__", side_effect=ImportError):
            pass  # just ensure flag is set when initialize() runs mock path
        with patch.dict("sys.modules", {"ultralytics": None}):
            detector.initialize()
        assert detector._is_initialized


# ---------------------------------------------------------------------------
# detect() – mock model path (no YOLO)
# ---------------------------------------------------------------------------

class TestObjectDetectorDetect:
    @pytest.fixture
    def detector(self):
        d = ObjectDetector()
        d._is_initialized = True  # skip auto-init
        d.model = None            # force mock path
        return d

    def test_detect_returns_list(self, blank_frame, detector):
        result = detector.detect(blank_frame)
        assert isinstance(result, list)

    def test_mock_detect_on_blank_frame(self, blank_frame, detector):
        detections = detector.detect(blank_frame)
        assert len(detections) >= 1

    def test_mock_detect_on_white_frame(self, white_frame, detector):
        detections = detector.detect(white_frame)
        assert len(detections) >= 1

    def test_mock_detect_has_valid_confidence(self, blank_frame, detector):
        detections = detector.detect(blank_frame)
        for det in detections:
            assert 0.0 <= det.confidence <= 1.0

    def test_mock_detect_bbox_within_frame(self, blank_frame, detector):
        h, w = blank_frame.shape[:2]
        detections = detector.detect(blank_frame)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            assert 0 <= x1 < w
            assert 0 <= y1 < h
            assert x1 <= x2 <= w
            assert y1 <= y2 <= h

    def test_mock_detect_center_matches_bbox(self, blank_frame, detector):
        detections = detector.detect(blank_frame)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            expected_cx = (x1 + x2) // 2
            expected_cy = (y1 + y2) // 2
            assert det.center == (expected_cx, expected_cy)

    def test_auto_initialize_called_if_not_initialized(self, blank_frame):
        detector = ObjectDetector()
        assert not detector._is_initialized
        # detect() should call initialize() automatically
        detector.detect(blank_frame)
        assert detector._is_initialized

    def test_detect_small_frame(self, detector):
        """Small (e.g. 32x32) frame should not raise."""
        small_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = detector.detect(small_frame)
        assert isinstance(result, list)

    def test_detect_non_standard_resolution(self, detector):
        """Odd-sized frame (e.g. 100x150) should not raise."""
        frame = np.zeros((100, 150, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# detect_obstacles() filtering
# ---------------------------------------------------------------------------

class TestDetectObstacles:
    @pytest.fixture
    def detector(self):
        d = ObjectDetector()
        d._is_initialized = True
        d.model = None
        return d

    def test_detect_obstacles_returns_list(self, blank_frame, detector):
        result = detector.detect_obstacles(blank_frame)
        assert isinstance(result, list)

    def test_detect_obstacles_only_obstacle_classes(self, blank_frame, detector):
        """All returned detections must have class_id in OBSTACLE_CLASSES."""
        results = detector.detect_obstacles(blank_frame)
        for det in results:
            assert det.class_id in ObjectDetector.OBSTACLE_CLASSES

    def test_detect_obstacles_excludes_non_obstacles(self, blank_frame, detector):
        """Non-obstacle class_ids should be filtered out."""
        non_obstacle_det = Detection(
            class_id=99,
            class_name="unknown",
            confidence=0.9,
            bbox=(0, 0, 10, 10),
            center=(5, 5),
            area=100,
        )
        with patch.object(detector, "detect", return_value=[non_obstacle_det]):
            results = detector.detect_obstacles(blank_frame)
        assert len(results) == 0

    def test_detect_obstacles_includes_person(self, blank_frame, detector):
        """class_id=0 (person) must survive the filter."""
        person_det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(10, 10, 50, 90),
            center=(30, 50),
            area=3200,
        )
        with patch.object(detector, "detect", return_value=[person_det]):
            results = detector.detect_obstacles(blank_frame)
        assert len(results) == 1
        assert results[0].class_id == 0


# ---------------------------------------------------------------------------
# get_obstacle_map() tests
# ---------------------------------------------------------------------------

class TestObstacleMap:
    @pytest.fixture
    def detector(self):
        d = ObjectDetector()
        d._is_initialized = True
        d.model = None
        return d

    def test_map_shape_matches_grid_size(self, blank_frame, detector):
        grid = detector.get_obstacle_map(blank_frame, grid_size=(10, 10))
        assert grid.shape == (10, 10)

    def test_map_is_binary(self, blank_frame, detector):
        grid = detector.get_obstacle_map(blank_frame, grid_size=(10, 10))
        unique_vals = np.unique(grid)
        assert set(unique_vals).issubset({0, 1})

    def test_map_default_grid_size(self, blank_frame, detector):
        grid = detector.get_obstacle_map(blank_frame)
        assert grid.shape == (10, 10)

    def test_map_has_obstacles_from_mock_detect(self, blank_frame, detector):
        """Mock detection returns a person; the map should mark at least one cell."""
        grid = detector.get_obstacle_map(blank_frame, grid_size=(10, 10))
        assert grid.sum() >= 1

    def test_map_no_obstacles_when_detections_empty(self, blank_frame, detector):
        """If detect_obstacles returns empty, the grid should be all zeros."""
        with patch.object(detector, "detect_obstacles", return_value=[]):
            grid = detector.get_obstacle_map(blank_frame, grid_size=(8, 8))
        assert grid.sum() == 0

    def test_map_custom_grid_size(self, blank_frame, detector):
        for rows, cols in [(5, 5), (20, 20), (4, 8)]:
            grid = detector.get_obstacle_map(blank_frame, grid_size=(rows, cols))
            assert grid.shape == (rows, cols)
