"""
Object Tracking Module for Drone AI

Provides multi-object tracking capabilities for following targets
and maintaining awareness of moving obstacles.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger


@dataclass
class TrackedObject:
    """Represents a tracked object with history."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)


class ObjectTracker:
    """
    Multi-object tracker using simple IoU-based association.
    
    Tracks objects across frames and estimates their velocities
    for prediction and collision avoidance.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize the tracker.
        
        Args:
            max_age: Maximum frames to keep a track without updates
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.frame_count = 0
        
        logger.info("ObjectTracker initialized")
    
    def update(self, detections: List[Tuple[int, int, int, int, str]]) -> List[TrackedObject]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, class_name) tuples
            
        Returns:
            List of confirmed tracked objects
        """
        self.frame_count += 1
        
        # Increment time since update for all tracks
        for track in self.tracks.values():
            track.time_since_update += 1
        
        if not detections:
            self._remove_stale_tracks()
            return self._get_confirmed_tracks()
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, det_idx in matched:
            self._update_track(track_id, detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])
        
        # Remove stale tracks
        self._remove_stale_tracks()
        
        return self._get_confirmed_tracks()
    
    def _match_detections(
        self, 
        detections: List[Tuple]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to existing tracks using IoU."""
        if not self.tracks:
            return [], list(range(len(detections))), []
        
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].bbox for tid in track_ids]
        det_boxes = [(d[0], d[1], d[2], d[3]) for d in detections]
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(track_boxes), len(det_boxes)))
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                iou_matrix[i, j] = self._compute_iou(tb, db)
        
        # Greedy matching
        matched = []
        matched_dets = set()
        matched_tracks = set()
        
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((track_ids[i], j))
            matched_tracks.add(i)
            matched_dets.add(j)
            
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_tracks]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _update_track(self, track_id: int, detection: Tuple):
        """Update an existing track with a new detection."""
        track = self.tracks[track_id]
        old_center = track.center
        
        x1, y1, x2, y2, class_name = detection
        new_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Estimate velocity
        velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )
        
        track.bbox = (x1, y1, x2, y2)
        track.center = new_center
        track.velocity = velocity
        track.hits += 1
        track.time_since_update = 0
        track.history.append(new_center)
        
        # Keep history limited
        if len(track.history) > 30:
            track.history = track.history[-30:]
    
    def _create_track(self, detection: Tuple):
        """Create a new track from a detection."""
        x1, y1, x2, y2, class_name = detection
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        track = TrackedObject(
            track_id=self.next_id,
            class_name=class_name,
            bbox=(x1, y1, x2, y2),
            center=center,
            history=[center]
        )
        
        self.tracks[self.next_id] = track
        self.next_id += 1
        
        logger.debug(f"Created new track {track.track_id} for {class_name}")
    
    def _remove_stale_tracks(self):
        """Remove tracks that haven't been updated recently."""
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if track.time_since_update > self.max_age
        ]
        
        for tid in stale_ids:
            logger.debug(f"Removing stale track {tid}")
            del self.tracks[tid]
    
    def _get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get tracks that have been confirmed (enough hits)."""
        return [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]
    
    def predict_positions(self, frames_ahead: int = 10) -> Dict[int, Tuple[int, int]]:
        """
        Predict future positions of tracked objects.
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Dictionary mapping track_id to predicted (x, y) position
        """
        predictions = {}
        
        for track in self._get_confirmed_tracks():
            pred_x = int(track.center[0] + track.velocity[0] * frames_ahead)
            pred_y = int(track.center[1] + track.velocity[1] * frames_ahead)
            predictions[track.track_id] = (pred_x, pred_y)
        
        return predictions
