"""Tests for the Visual SLAM module."""

import numpy as np
import pytest

from src.vision.slam import (
    CameraModel,
    FeatureTracker,
    PoseEstimator,
    VisualSLAM,
    SLAMState,
    Keyframe,
    Landmark,
    _MockKeypoint,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_camera(width: int = 640, height: int = 480) -> CameraModel:
    return CameraModel.from_fov(width=width, height=height, hfov_deg=90.0)


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a random frame (consistent with FeatureTracker seeding)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _moving_frame(h: int = 480, w: int = 640, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# CameraModel
# ─────────────────────────────────────────────────────────────

class TestCameraModel:
    def test_intrinsic_matrix_shape(self):
        cam = _make_camera()
        assert cam.K.shape == (3, 3)

    def test_intrinsic_matrix_values(self):
        cam = _make_camera(width=640, height=480)
        assert cam.K[0, 0] == pytest.approx(cam.fx)
        assert cam.K[1, 1] == pytest.approx(cam.fy)
        assert cam.K[0, 2] == pytest.approx(cam.cx)
        assert cam.K[1, 2] == pytest.approx(cam.cy)

    def test_from_fov_cx_cy(self):
        cam = CameraModel.from_fov(width=640, height=480, hfov_deg=90.0)
        assert cam.cx == pytest.approx(320.0)
        assert cam.cy == pytest.approx(240.0)

    def test_from_fov_focal_length(self):
        cam = CameraModel.from_fov(width=640, height=480, hfov_deg=90.0)
        assert cam.fx == pytest.approx(320.0, rel=1e-3)

    def test_distortion_default_zeros(self):
        cam = _make_camera()
        np.testing.assert_array_equal(cam.distortion, np.zeros(5))


# ─────────────────────────────────────────────────────────────
# FeatureTracker
# ─────────────────────────────────────────────────────────────

class TestFeatureTracker:
    def test_detect_returns_keypoints_and_descriptors(self):
        tracker = FeatureTracker()
        frame = _blank_frame()
        kps, descs = tracker.detect_and_compute(frame)
        assert isinstance(kps, list)
        assert descs is not None
        assert len(kps) > 0

    def test_descriptors_shape(self):
        tracker = FeatureTracker()
        frame = _blank_frame()
        kps, descs = tracker.detect_and_compute(frame)
        assert descs.ndim == 2
        assert descs.shape[1] == 32  # ORB / mock always use 32-byte descriptors

    def test_max_features_respected(self):
        tracker = FeatureTracker(max_features=20)
        frame = _blank_frame()
        kps, descs = tracker.detect_and_compute(frame)
        assert len(kps) <= 20

    def test_match_returns_list(self):
        tracker = FeatureTracker()
        frame = _blank_frame()
        _, descs = tracker.detect_and_compute(frame)
        matches = tracker.match(descs, descs)
        assert isinstance(matches, list)

    def test_match_identical_descriptors(self):
        tracker = FeatureTracker()
        frame = _blank_frame()
        _, descs = tracker.detect_and_compute(frame)
        matches = tracker.match(descs, descs)
        assert len(matches) > 0

    def test_match_none_descriptors(self):
        tracker = FeatureTracker()
        _, descs = tracker.detect_and_compute(_blank_frame())
        assert tracker.match(None, descs) == []
        assert tracker.match(descs, None) == []

    def test_match_empty_descriptors(self):
        tracker = FeatureTracker()
        empty = np.zeros((0, 32), dtype=np.uint8)
        _, descs = tracker.detect_and_compute(_blank_frame())
        assert tracker.match(empty, descs) == []


# ─────────────────────────────────────────────────────────────
# PoseEstimator
# ─────────────────────────────────────────────────────────────

class TestPoseEstimator:
    def test_estimate_too_few_points(self):
        cam = _make_camera()
        estimator = PoseEstimator(cam)
        pts = np.random.rand(4, 2).astype(np.float32) * 100
        R, t, mask = estimator.estimate_relative_pose(pts, pts)
        assert R is None
        assert t is None

    def test_estimate_returns_rotation_and_translation(self):
        cam = _make_camera()
        estimator = PoseEstimator(cam)
        rng = np.random.default_rng(7)
        pts1 = rng.uniform(100, 540, (20, 2)).astype(np.float32)
        pts2 = pts1 + rng.uniform(-5, 5, pts1.shape).astype(np.float32)
        R, t, mask = estimator.estimate_relative_pose(pts1, pts2)
        if R is not None:
            assert R.shape == (3, 3)
            assert t.shape == (3,)

    def test_triangulate_points_shape(self):
        cam = _make_camera()
        estimator = PoseEstimator(cam)
        pts1 = np.array([[100.0, 200.0], [300.0, 150.0], [400.0, 300.0]], dtype=np.float32)
        pts2 = pts1 + 5.0
        R = np.eye(3)
        t = np.array([1.0, 0.0, 0.0])
        pts3d = estimator.triangulate_points(pts1, pts2, R, t)
        assert pts3d.shape[1] == 3

    def test_triangulate_empty_returns_empty(self):
        cam = _make_camera()
        estimator = PoseEstimator(cam)
        pts3d = estimator.triangulate_points(
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.eye(3),
            np.zeros(3),
        )
        assert pts3d.shape == (0, 3)


# ─────────────────────────────────────────────────────────────
# VisualSLAM
# ─────────────────────────────────────────────────────────────

class TestVisualSLAM:
    def test_init(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        assert slam.keyframes == []
        assert slam.landmarks == []

    def test_process_first_frame_initialises(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        frame = _blank_frame()
        state = slam.process_frame(frame, timestamp=0.0)
        assert isinstance(state, SLAMState)
        assert state.num_keyframes >= 1

    def test_state_fields_present(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        state = slam.process_frame(_blank_frame(), timestamp=0.0)
        assert state.position.shape == (3,)
        assert state.orientation.shape == (3, 3)
        assert isinstance(state.tracking_ok, bool)
        assert isinstance(state.loop_detected, bool)

    def test_process_multiple_frames(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        for i in range(5):
            frame = _moving_frame(seed=i)
            state = slam.process_frame(frame, timestamp=float(i))
        assert state.num_keyframes >= 1

    def test_reset_clears_state(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        for i in range(3):
            slam.process_frame(_moving_frame(seed=i), timestamp=float(i))
        slam.reset()
        assert slam.keyframes == []
        assert slam.landmarks == []
        np.testing.assert_array_equal(slam._global_t, np.zeros(3))
        np.testing.assert_array_equal(slam._global_R, np.eye(3))

    def test_get_trajectory_before_processing(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        assert slam.get_trajectory() == []

    def test_get_trajectory_after_processing(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        for i in range(3):
            slam.process_frame(_moving_frame(seed=i), timestamp=float(i))
        traj = slam.get_trajectory()
        assert isinstance(traj, list)
        for pt in traj:
            assert pt.shape == (3,)

    def test_get_map_points_shape(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        for i in range(3):
            slam.process_frame(_moving_frame(seed=i), timestamp=float(i))
        pts = slam.get_map_points()
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_get_map_points_empty_before_processing(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        pts = slam.get_map_points()
        assert pts.shape == (0, 3)

    def test_max_landmarks_respected(self):
        cam = _make_camera()
        slam = VisualSLAM(cam, max_landmarks=10)
        for i in range(10):
            slam.process_frame(_moving_frame(seed=i), timestamp=float(i))
        assert len(slam.landmarks) <= 10

    def test_loop_closure_not_detected_few_frames(self):
        cam = _make_camera()
        slam = VisualSLAM(cam)
        states = []
        for i in range(4):
            s = slam.process_frame(_moving_frame(seed=i), timestamp=float(i))
            states.append(s)
        # With only 4 frames loop detection is suppressed (needs > 5 keyframes)
        assert all(not s.loop_detected for s in states)
