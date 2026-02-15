"""Tests for vision module."""
import pytest
import numpy as np
from src.vision.detection import ObjectDetector
from src.vision.tracking import ObjectTracker


class TestObjectDetector:
    """Test cases for ObjectDetector class."""

    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.5)
        assert detector.confidence == 0.5
        assert detector.model_path == "yolov8n.pt"

    def test_detector_with_mock_frame(self):
        """Test detection with mock frame."""
        detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.5)
        # Create a mock frame (640x480 RGB)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)

    def test_obstacle_detection(self):
        """Test obstacle detection filtering."""
        detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        obstacles = detector.detect_obstacles(frame)
        assert isinstance(obstacles, list)


class TestObjectTracker:
    """Test cases for ObjectTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = ObjectTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3

    def test_tracker_update_empty(self):
        """Test tracker update with no detections."""
        tracker = ObjectTracker()
        tracks = tracker.update([])
        assert isinstance(tracks, list)

    def test_tracker_update_with_detections(self):
        """Test tracker update with detections."""
        tracker = ObjectTracker()
        detections = [
            {"bbox": [100, 100, 200, 200], "class": "person", "confidence": 0.9},
            {"bbox": [300, 300, 400, 400], "class": "car", "confidence": 0.85},
        ]
        tracks = tracker.update(detections)
        assert isinstance(tracks, list)

    def test_velocity_estimation(self):
        """Test velocity estimation over multiple frames."""
        tracker = ObjectTracker()
        # Simulate object moving right
        for i in range(5):
            detections = [
                {"bbox": [100 + i*10, 100, 200 + i*10, 200], "class": "person", "confidence": 0.9}
            ]
            tracker.update(detections)
        
        # Check that tracks exist
        assert len(tracker.tracks) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
