"""
Visual SLAM Module for Drone AI

Implements a lightweight monocular Visual Simultaneous Localization and
Mapping (SLAM) pipeline for GPS-denied environments.  The pipeline combines:

  - Feature extraction and matching (ORB descriptors via OpenCV)
  - Pose estimation via Essential-matrix decomposition
  - A local map of 3-D landmarks
  - Loop-closure detection via a bag-of-words vocabulary comparison

The implementation uses only numpy and opencv-python so it runs on any
platform without additional deep-learning frameworks.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from loguru import logger


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class Keyframe:
    """A camera frame selected as a SLAM keyframe."""
    frame_id: int
    keypoints: List  # cv2.KeyPoint list
    descriptors: np.ndarray
    pose: np.ndarray  # 4×4 homogeneous transformation (camera → world)
    timestamp: float


@dataclass
class Landmark:
    """A 3-D map point triangulated from two or more keyframes."""
    landmark_id: int
    position: np.ndarray   # [X, Y, Z] in world frame
    descriptor: np.ndarray
    observed_by: List[int] = field(default_factory=list)  # keyframe IDs


@dataclass
class SLAMState:
    """Current state of the SLAM system."""
    position: np.ndarray       # [x, y, z] in world frame
    orientation: np.ndarray    # 3×3 rotation matrix (camera → world)
    num_keyframes: int
    num_landmarks: int
    tracking_ok: bool
    loop_detected: bool


# ─────────────────────────────────────────────────────────────
# Camera model
# ─────────────────────────────────────────────────────────────

@dataclass
class CameraModel:
    """Pinhole camera intrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float = 90.0,
    ) -> "CameraModel":
        """Convenience constructor from horizontal FOV."""
        fx = width / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


# ─────────────────────────────────────────────────────────────
# Feature tracker
# ─────────────────────────────────────────────────────────────

class FeatureTracker:
    """
    ORB feature extractor and BFMatcher-based descriptor matcher.

    Falls back to a simple random-feature mock when OpenCV is not
    available or when the image has no detectable features.
    """

    def __init__(
        self,
        max_features: int = 500,
        match_ratio: float = 0.75,
    ):
        self.max_features = max_features
        self.match_ratio = match_ratio
        self._orb = None
        self._matcher = None
        self._init_opencv()

    def _init_opencv(self):
        try:
            import cv2
            self._orb = cv2.ORB_create(nfeatures=self.max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            logger.debug("FeatureTracker: using OpenCV ORB")
        except ImportError:
            logger.warning("FeatureTracker: opencv-python not found, using mock features")

    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect keypoints and compute ORB descriptors.

        Returns:
            (keypoints, descriptors) – descriptors is None if no features found.
        """
        if self._orb is not None:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            kps, descs = self._orb.detectAndCompute(gray, None)
            return list(kps) if kps is not None else [], descs
        return self._mock_features(image)

    def _mock_features(
        self, image: np.ndarray
    ) -> Tuple[List, np.ndarray]:
        """Generate deterministic pseudo-random mock features."""
        rng = np.random.default_rng(seed=int(image.mean() * 1000) % (2**31))
        n = min(self.max_features, 50)
        h, w = image.shape[:2]
        kps = [
            _MockKeypoint(
                pt=(float(rng.integers(0, w)), float(rng.integers(0, h))),
                size=8.0,
                angle=float(rng.uniform(0, 360)),
            )
            for _ in range(n)
        ]
        descs = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
        return kps, descs

    def match(
        self,
        descs1: np.ndarray,
        descs2: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Match descriptors using Lowe's ratio test.

        Returns:
            List of (idx1, idx2) matched pairs.
        """
        if descs1 is None or descs2 is None:
            return []
        if len(descs1) == 0 or len(descs2) == 0:
            return []

        if self._matcher is not None:
            raw_matches = self._matcher.knnMatch(descs1, descs2, k=2)
            good = []
            for m_pair in raw_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self.match_ratio * n.distance:
                        good.append((m.queryIdx, m.trainIdx))
            return good
        return self._mock_match(descs1, descs2)

    def _mock_match(
        self,
        descs1: np.ndarray,
        descs2: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Hamming-distance nearest-neighbour mock matcher."""
        pairs = []
        n1, n2 = len(descs1), len(descs2)
        for i in range(min(n1, 20)):
            dists = np.count_nonzero(descs1[i] != descs2[:n2], axis=1)
            j = int(np.argmin(dists))
            pairs.append((i, j))
        return pairs


class _MockKeypoint:
    """Minimal stand-in for cv2.KeyPoint used when OpenCV is absent."""
    def __init__(self, pt: Tuple[float, float], size: float, angle: float):
        self.pt = pt
        self.size = size
        self.angle = angle


# ─────────────────────────────────────────────────────────────
# Pose estimator
# ─────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Relative pose estimation from matched 2-D point correspondences.

    Uses the Essential matrix (5-point algorithm via RANSAC when OpenCV
    is available) or a simplified SVD-based homography otherwise.
    """

    def __init__(self, camera: CameraModel):
        self.camera = camera

    def estimate_relative_pose(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Estimate the relative rotation R and translation t between two frames.

        Args:
            pts1: (N, 2) matched points from frame 1.
            pts2: (N, 2) matched points from frame 2.

        Returns:
            (R, t, inlier_mask) where R is 3×3, t is (3,), mask is boolean array.
            Returns (None, None, empty_mask) when estimation fails.
        """
        if len(pts1) < 5:
            return None, None, np.array([], dtype=bool)

        try:
            import cv2
            E, mask = cv2.findEssentialMat(
                pts1, pts2,
                cameraMatrix=self.camera.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
            if E is None or mask is None:
                return None, None, np.array([], dtype=bool)
            _, R, t, mask2 = cv2.recoverPose(E, pts1, pts2, self.camera.K, mask=mask)
            return R, t.ravel(), mask2.ravel().astype(bool)
        except (ImportError, Exception):
            return self._svd_estimate(pts1, pts2)

    def _svd_estimate(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Simplified planar homography-based pose estimate (fallback)."""
        try:
            # Normalise points
            K = self.camera.K
            pts1_n = (pts1 - np.array([K[0, 2], K[1, 2]])) / np.array([K[0, 0], K[1, 1]])
            pts2_n = (pts2 - np.array([K[0, 2], K[1, 2]])) / np.array([K[0, 0], K[1, 1]])

            # Compute centroid-normalised transformation via SVD
            c1 = pts1_n.mean(axis=0)
            c2 = pts2_n.mean(axis=0)
            H = (pts2_n - c2).T @ (pts1_n - c1)
            U, _, Vt = np.linalg.svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = U @ Vt
            t = c2 - R[:2, :2] @ c1
            R3 = np.eye(3)
            R3[:2, :2] = R
            t3 = np.array([t[0], t[1], 0.0])
            mask = np.ones(len(pts1), dtype=bool)
            return R3, t3, mask
        except Exception as exc:
            logger.warning(f"PoseEstimator: SVD fallback failed – {exc}")
            return None, None, np.array([], dtype=bool)

    def triangulate_points(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Triangulate 3-D points from two views.

        Args:
            pts1: (N, 2) points in frame 1.
            pts2: (N, 2) points in frame 2.
            R: 3×3 rotation matrix (frame 1 → frame 2).
            t: Translation vector (3,).

        Returns:
            (N, 3) array of 3-D points in frame-1 coordinates.
        """
        K = self.camera.K
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t.reshape(3, 1)])

        try:
            import cv2
            pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float32),
                                          pts2.T.astype(np.float32))
            pts3d = (pts4d[:3] / pts4d[3]).T
            return pts3d
        except (ImportError, Exception):
            return self._dlt_triangulate(P1, P2, pts1, pts2)

    def _dlt_triangulate(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> np.ndarray:
        """Direct Linear Transform triangulation (fallback)."""
        pts3d = []
        for p1, p2 in zip(pts1, pts2):
            A = np.array([
                p1[0] * P1[2] - P1[0],
                p1[1] * P1[2] - P1[1],
                p2[0] * P2[2] - P2[0],
                p2[1] * P2[2] - P2[1],
            ])
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            pts3d.append(X[:3] / X[3])
        return np.array(pts3d) if pts3d else np.zeros((0, 3))


# ─────────────────────────────────────────────────────────────
# Visual SLAM
# ─────────────────────────────────────────────────────────────

class VisualSLAM:
    """
    Lightweight monocular Visual SLAM for autonomous drone navigation.

    Pipeline per frame:
      1. Extract ORB features.
      2. Match against the previous keyframe.
      3. Estimate relative pose via Essential matrix.
      4. Triangulate new map points.
      5. Detect loop closures via descriptor voting.
      6. Accumulate global pose.

    The system outputs a ``SLAMState`` on every processed frame
    and keeps a list of ``Landmark`` objects that form the sparse map.
    """

    def __init__(
        self,
        camera: CameraModel,
        min_matches: int = 8,
        keyframe_min_matches: int = 20,
        max_landmarks: int = 5000,
    ):
        """
        Initialise the SLAM system.

        Args:
            camera: Camera intrinsics.
            min_matches: Minimum feature matches required to estimate pose.
            keyframe_min_matches: New keyframe created when match count drops
                                  below this threshold.
            max_landmarks: Cap on the number of landmarks kept in the map.
        """
        self.camera = camera
        self.min_matches = min_matches
        self.keyframe_min_matches = keyframe_min_matches
        self.max_landmarks = max_landmarks

        self.tracker = FeatureTracker()
        self.pose_estimator = PoseEstimator(camera)

        self.keyframes: List[Keyframe] = []
        self.landmarks: List[Landmark] = []
        self._next_landmark_id = 0

        # Accumulated global pose (world frame)
        self._global_R = np.eye(3)
        self._global_t = np.zeros(3)

        self._frame_id = 0
        self._tracking_ok = False

        logger.info("VisualSLAM initialised")

    def process_frame(
        self, image: np.ndarray, timestamp: float = 0.0
    ) -> SLAMState:
        """
        Process one image frame and update the map/pose estimate.

        Args:
            image: BGR or grayscale image as numpy array.
            timestamp: Frame timestamp in seconds.

        Returns:
            SLAMState with current pose and map statistics.
        """
        self._frame_id += 1
        kps, descs = self.tracker.detect_and_compute(image)
        loop_detected = False

        if not self.keyframes or descs is None or len(kps) < self.min_matches:
            # Initialise or mark as lost
            if descs is not None and len(kps) >= self.min_matches:
                self._add_keyframe(kps, descs, np.eye(4), timestamp)
                self._tracking_ok = True
            else:
                self._tracking_ok = False
            return self._build_state(loop_detected=False)

        # ── Match against latest keyframe ──────────────────────────
        ref_kf = self.keyframes[-1]
        matches = self.tracker.match(ref_kf.descriptors, descs)

        if len(matches) < self.min_matches:
            self._tracking_ok = False
            return self._build_state(loop_detected=False)

        pts_ref = np.array([ref_kf.keypoints[m[0]].pt for m in matches],
                           dtype=np.float32)
        pts_cur = np.array([kps[m[1]].pt for m in matches], dtype=np.float32)

        # ── Pose estimation ────────────────────────────────────────
        R, t, inliers = self.pose_estimator.estimate_relative_pose(pts_ref, pts_cur)

        if R is None or inliers.sum() < self.min_matches:
            self._tracking_ok = False
            return self._build_state(loop_detected=False)

        # Accumulate pose
        self._global_t = self._global_t + self._global_R @ t
        self._global_R = R @ self._global_R

        # ── Triangulate new landmarks ──────────────────────────────
        inlier_pts_ref = pts_ref[inliers]
        inlier_pts_cur = pts_cur[inliers]
        if len(inlier_pts_ref) >= 4:
            pts3d = self.pose_estimator.triangulate_points(
                inlier_pts_ref, inlier_pts_cur, R, t
            )
            self._add_landmarks(pts3d, descs, inliers, matches)

        # ── Decide on new keyframe ─────────────────────────────────
        if len(matches) < self.keyframe_min_matches:
            pose4x4 = self._make_pose4x4(self._global_R, self._global_t)
            self._add_keyframe(kps, descs, pose4x4, timestamp)

        # ── Loop closure check ─────────────────────────────────────
        if len(self.keyframes) > 5:
            loop_detected = self._detect_loop(descs)

        self._tracking_ok = True
        return self._build_state(loop_detected=loop_detected)

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _add_keyframe(
        self,
        kps: List,
        descs: np.ndarray,
        pose: np.ndarray,
        timestamp: float,
    ):
        """Store a new keyframe."""
        kf = Keyframe(
            frame_id=self._frame_id,
            keypoints=kps,
            descriptors=descs,
            pose=pose,
            timestamp=timestamp,
        )
        self.keyframes.append(kf)
        logger.debug(f"VisualSLAM: keyframe {self._frame_id} added (total={len(self.keyframes)})")

    def _add_landmarks(
        self,
        pts3d: np.ndarray,
        descs: np.ndarray,
        inliers: np.ndarray,
        matches: List[Tuple[int, int]],
    ):
        """Add newly triangulated 3-D points to the map."""
        if len(pts3d) == 0:
            return
        inlier_match_indices = np.where(inliers)[0]

        for k, pt in enumerate(pts3d):
            if k >= len(inlier_match_indices):
                break
            if len(self.landmarks) >= self.max_landmarks:
                break
            match_idx = inlier_match_indices[k]
            if match_idx >= len(matches):
                break
            desc_idx = matches[match_idx][1]
            if desc_idx >= len(descs):
                break

            lm = Landmark(
                landmark_id=self._next_landmark_id,
                position=pt.copy(),
                descriptor=descs[desc_idx].copy(),
                observed_by=[self.keyframes[-1].frame_id if self.keyframes else 0],
            )
            self.landmarks.append(lm)
            self._next_landmark_id += 1

    def _detect_loop(self, descs: np.ndarray) -> bool:
        """
        Simple loop-closure detection via descriptor voting.

        Counts how many descriptors in the current frame are close to
        descriptors from older keyframes (skipping the 3 most recent).
        """
        if descs is None or len(descs) == 0 or len(self.keyframes) < 6:
            return False

        vote_threshold = max(5, self.min_matches)
        candidate_kfs = self.keyframes[:-3]  # exclude recent

        for kf in candidate_kfs:
            if kf.descriptors is None:
                continue
            matches = self.tracker.match(descs, kf.descriptors)
            if len(matches) >= vote_threshold:
                logger.info(
                    f"VisualSLAM: loop closure detected – frame {self._frame_id} "
                    f"→ keyframe {kf.frame_id} ({len(matches)} matches)"
                )
                return True
        return False

    @staticmethod
    def _make_pose4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Assemble a 4×4 homogeneous pose matrix."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _build_state(self, loop_detected: bool) -> SLAMState:
        """Build a SLAMState from the current system state."""
        return SLAMState(
            position=self._global_t.copy(),
            orientation=self._global_R.copy(),
            num_keyframes=len(self.keyframes),
            num_landmarks=len(self.landmarks),
            tracking_ok=self._tracking_ok,
            loop_detected=loop_detected,
        )

    def get_trajectory(self) -> List[np.ndarray]:
        """Return the list of keyframe positions in world frame."""
        return [kf.pose[:3, 3].copy() for kf in self.keyframes]

    def get_map_points(self) -> np.ndarray:
        """Return all landmark positions as an (N, 3) array."""
        if not self.landmarks:
            return np.zeros((0, 3))
        return np.array([lm.position for lm in self.landmarks])

    def reset(self):
        """Reset the SLAM system to its initial state."""
        self.keyframes.clear()
        self.landmarks.clear()
        self._next_landmark_id = 0
        self._global_R = np.eye(3)
        self._global_t = np.zeros(3)
        self._frame_id = 0
        self._tracking_ok = False
        logger.info("VisualSLAM reset")
