"""
Tests for vision/tracking.py

Covers:
- TrackedObject dataclass construction
- ObjectTracker initialization
- IoU computation (known values)
- Creating new tracks from detections
- Updating existing tracks
- Track age / time_since_update mechanics
- Stale track removal (max_age)
- Confirmed-track filtering (min_hits)
- Position prediction
- Lost-track handling (no detections provided)
- ID assignment uniqueness
"""
import numpy as np
import pytest

from vision.tracking import TrackedObject, ObjectTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detection(x1, y1, x2, y2, cls="person"):
    """Return a detection tuple as expected by ObjectTracker.update()."""
    return (x1, y1, x2, y2, cls)


# ---------------------------------------------------------------------------
# TrackedObject dataclass
# ---------------------------------------------------------------------------

class TestTrackedObjectDataclass:
    def test_basic_construction(self):
        obj = TrackedObject(
            track_id=0,
            class_name="person",
            bbox=(10, 20, 60, 80),
            center=(35, 50),
        )
        assert obj.track_id == 0
        assert obj.class_name == "person"
        assert obj.bbox == (10, 20, 60, 80)
        assert obj.center == (35, 50)

    def test_default_velocity(self):
        obj = TrackedObject(track_id=1, class_name="car", bbox=(0, 0, 10, 10), center=(5, 5))
        assert obj.velocity == (0.0, 0.0)

    def test_default_age_and_hits(self):
        obj = TrackedObject(track_id=2, class_name="car", bbox=(0, 0, 10, 10), center=(5, 5))
        assert obj.age == 0
        assert obj.hits == 1
        assert obj.time_since_update == 0

    def test_history_initialized_empty_by_default(self):
        obj = TrackedObject(track_id=3, class_name="car", bbox=(0, 0, 10, 10), center=(5, 5))
        assert obj.history == []


# ---------------------------------------------------------------------------
# ObjectTracker initialization
# ---------------------------------------------------------------------------

class TestObjectTrackerInit:
    def test_defaults(self):
        tracker = ObjectTracker()
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == pytest.approx(0.3)

    def test_custom_params(self):
        tracker = ObjectTracker(max_age=10, min_hits=1, iou_threshold=0.5)
        assert tracker.max_age == 10
        assert tracker.min_hits == 1
        assert tracker.iou_threshold == pytest.approx(0.5)

    def test_initial_state_empty(self):
        tracker = ObjectTracker()
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 0
        assert tracker.frame_count == 0


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

class TestIoUComputation:
    @pytest.fixture
    def tracker(self):
        return ObjectTracker()

    def test_identical_boxes_iou_is_one(self, tracker):
        box = (0, 0, 100, 100)
        assert tracker._compute_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_zero(self, tracker):
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert tracker._compute_iou(box1, box2) == pytest.approx(0.0)

    def test_half_overlap_horizontal(self, tracker):
        # box1 = [0,0,10,10], box2 = [5,0,15,10]
        # intersection = 5*10=50, union=200-50=150
        box1 = (0, 0, 10, 10)
        box2 = (5, 0, 15, 10)
        iou = tracker._compute_iou(box1, box2)
        assert iou == pytest.approx(50 / 150, rel=1e-3)

    def test_contained_box_iou(self, tracker):
        # box2 fully inside box1
        box1 = (0, 0, 100, 100)
        box2 = (25, 25, 75, 75)
        # intersection = 50*50=2500, union=10000+2500-2500=10000
        iou = tracker._compute_iou(box1, box2)
        assert iou == pytest.approx(2500 / 10000)

    def test_iou_symmetric(self, tracker):
        box1 = (0, 0, 40, 40)
        box2 = (20, 20, 60, 60)
        assert tracker._compute_iou(box1, box2) == pytest.approx(
            tracker._compute_iou(box2, box1)
        )

    def test_touching_edges_no_overlap(self, tracker):
        box1 = (0, 0, 10, 10)
        box2 = (10, 0, 20, 10)
        assert tracker._compute_iou(box1, box2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Creating new tracks
# ---------------------------------------------------------------------------

class TestTrackCreation:
    def test_first_detection_creates_track(self):
        tracker = ObjectTracker(min_hits=1)
        detections = [make_detection(0, 0, 50, 50)]
        tracks = tracker.update(detections)
        assert len(tracker.tracks) == 1

    def test_track_id_starts_at_zero(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([make_detection(0, 0, 50, 50)])
        assert 0 in tracker.tracks

    def test_multiple_detections_create_multiple_tracks(self):
        tracker = ObjectTracker(min_hits=1)
        dets = [
            make_detection(0, 0, 50, 50),
            make_detection(200, 200, 250, 250),
        ]
        tracker.update(dets)
        assert len(tracker.tracks) == 2

    def test_track_ids_are_unique(self):
        tracker = ObjectTracker(min_hits=1)
        for i in range(5):
            tracker.update([make_detection(i * 100, 0, i * 100 + 50, 50)])
        ids = list(tracker.tracks.keys())
        assert len(ids) == len(set(ids))

    def test_new_track_class_name_stored(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([make_detection(0, 0, 50, 50, "car")])
        track = list(tracker.tracks.values())[0]
        assert track.class_name == "car"

    def test_new_track_bbox_stored(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([make_detection(10, 20, 60, 80)])
        track = list(tracker.tracks.values())[0]
        assert track.bbox == (10, 20, 60, 80)

    def test_new_track_center_computed_correctly(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([make_detection(0, 0, 100, 50)])
        track = list(tracker.tracks.values())[0]
        assert track.center == (50, 25)


# ---------------------------------------------------------------------------
# Updating existing tracks
# ---------------------------------------------------------------------------

class TestTrackUpdates:
    def test_same_detection_increments_hits(self):
        tracker = ObjectTracker(min_hits=1)
        det = make_detection(0, 0, 50, 50)
        tracker.update([det])
        initial_hits = list(tracker.tracks.values())[0].hits
        tracker.update([det])
        updated_hits = list(tracker.tracks.values())[0].hits
        assert updated_hits == initial_hits + 1

    def test_update_resets_time_since_update(self):
        tracker = ObjectTracker(min_hits=1)
        det = make_detection(0, 0, 50, 50)
        tracker.update([det])
        # Now update again — time_since_update should be 0
        tracker.update([det])
        track = list(tracker.tracks.values())[0]
        assert track.time_since_update == 0

    def test_velocity_estimated_from_movement(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([make_detection(0, 0, 50, 50)])   # center=(25,25)
        tracker.update([make_detection(10, 10, 60, 60)]) # center=(35,35)
        track = list(tracker.tracks.values())[0]
        assert track.velocity == (10, 10)

    def test_history_grows_with_updates(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([make_detection(0, 0, 50, 50)])
        tracker.update([make_detection(10, 10, 60, 60)])
        track = list(tracker.tracks.values())[0]
        assert len(track.history) >= 1  # history appended on each update

    def test_history_capped_at_30(self):
        tracker = ObjectTracker(min_hits=1)
        # Feed 35 consecutive updates that match the same track
        tracker.update([make_detection(0, 0, 50, 50)])
        for i in range(35):
            tracker.update([make_detection(i, i, i + 50, i + 50)])
        track = list(tracker.tracks.values())[0]
        assert len(track.history) <= 30


# ---------------------------------------------------------------------------
# Stale track removal
# ---------------------------------------------------------------------------

class TestStaleTrackRemoval:
    def test_track_removed_after_max_age(self):
        tracker = ObjectTracker(max_age=3, min_hits=1)
        tracker.update([make_detection(0, 0, 50, 50)])
        # Feed empty detections for max_age+1 frames
        for _ in range(4):
            tracker.update([])
        assert len(tracker.tracks) == 0

    def test_track_survives_within_max_age(self):
        tracker = ObjectTracker(max_age=5, min_hits=1)
        tracker.update([make_detection(0, 0, 50, 50)])
        # Feed 4 empty frames (still within max_age=5)
        for _ in range(4):
            tracker.update([])
        assert len(tracker.tracks) == 1

    def test_no_detections_increments_time_since_update(self):
        tracker = ObjectTracker(max_age=10, min_hits=1)
        tracker.update([make_detection(0, 0, 50, 50)])
        tracker.update([])
        track = list(tracker.tracks.values())[0]
        assert track.time_since_update >= 1


# ---------------------------------------------------------------------------
# Confirmed-track filtering (min_hits)
# ---------------------------------------------------------------------------

class TestConfirmedTracks:
    def test_new_track_not_confirmed_immediately(self):
        tracker = ObjectTracker(min_hits=3)
        confirmed = tracker.update([make_detection(0, 0, 50, 50)])
        # Only 1 hit so far — not confirmed
        assert len(confirmed) == 0

    def test_track_confirmed_after_min_hits(self):
        tracker = ObjectTracker(min_hits=3)
        det = make_detection(0, 0, 50, 50)
        for _ in range(3):
            confirmed = tracker.update([det])
        assert len(confirmed) == 1

    def test_min_hits_one_confirms_immediately(self):
        tracker = ObjectTracker(min_hits=1)
        confirmed = tracker.update([make_detection(0, 0, 50, 50)])
        assert len(confirmed) == 1


# ---------------------------------------------------------------------------
# Position prediction
# ---------------------------------------------------------------------------

class TestPositionPrediction:
    def test_predict_returns_dict(self):
        tracker = ObjectTracker(min_hits=1)
        det = make_detection(0, 0, 50, 50)
        tracker.update([det])
        tracker.update([make_detection(10, 10, 60, 60)])
        preds = tracker.predict_positions(frames_ahead=5)
        assert isinstance(preds, dict)

    def test_predict_uses_velocity(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([make_detection(0, 0, 50, 50)])   # center=(25,25)
        tracker.update([make_detection(10, 10, 60, 60)]) # center=(35,35), velocity=(10,10)
        preds = tracker.predict_positions(frames_ahead=3)
        track_id = list(tracker.tracks.keys())[0]
        # predicted = (35+10*3, 35+10*3) = (65, 65)
        assert track_id in preds
        px, py = preds[track_id]
        assert px == 65
        assert py == 65

    def test_predict_no_confirmed_tracks_returns_empty(self):
        tracker = ObjectTracker(min_hits=5)
        tracker.update([make_detection(0, 0, 50, 50)])
        preds = tracker.predict_positions()
        assert len(preds) == 0

    def test_predict_zero_velocity_returns_current_center(self):
        tracker = ObjectTracker(min_hits=1)
        det = make_detection(0, 0, 50, 50)  # center=(25,25)
        tracker.update([det])
        tracker.update([det])  # same position → velocity=(0,0)
        preds = tracker.predict_positions(frames_ahead=10)
        track_id = list(tracker.tracks.keys())[0]
        px, py = preds[track_id]
        assert px == 25
        assert py == 25


# ---------------------------------------------------------------------------
# frame_count bookkeeping
# ---------------------------------------------------------------------------

class TestFrameCount:
    def test_frame_count_increments(self):
        tracker = ObjectTracker()
        assert tracker.frame_count == 0
        tracker.update([])
        assert tracker.frame_count == 1
        tracker.update([])
        assert tracker.frame_count == 2

    def test_frame_count_with_detections(self):
        tracker = ObjectTracker()
        for _ in range(5):
            tracker.update([make_detection(0, 0, 50, 50)])
        assert tracker.frame_count == 5
