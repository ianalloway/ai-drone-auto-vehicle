"""
Object Detection Module for Drone AI

Provides real-time object detection using YOLOv8 for obstacle identification,
landing zone detection, and target tracking.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class Detection:
    """Represents a detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int


class ObjectDetector:
    """
    Real-time object detection using YOLOv8.
    
    Supports detection of obstacles, people, vehicles, and other objects
    relevant to autonomous drone navigation.
    """
    
    # Default classes relevant to drone navigation
    OBSTACLE_CLASSES = {
        0: "person",
        1: "bicycle", 
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        14: "bird",
        15: "cat",
        16: "dog",
        64: "potted plant",
        67: "dining table",
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._is_initialized = False
        
        logger.info(f"ObjectDetector initialized with model: {model_path}")
    
    def initialize(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self._is_initialized = True
            logger.info("YOLO model loaded successfully")
            return True
        except ImportError:
            logger.warning("ultralytics not installed, using mock detector")
            self._is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform object detection on a frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        if not self._is_initialized:
            self.initialize()
        
        detections = []
        
        if self.model is None:
            # Mock detection for testing without YOLO
            return self._mock_detect(frame)
        
        try:
            results = self.model(frame, conf=self.confidence_threshold, device=self.device)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        area=area
                    ))
            
            logger.debug(f"Detected {len(detections)} objects")
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        return detections
    
    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """Generate mock detections for testing."""
        h, w = frame.shape[:2]
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.85,
                bbox=(w//4, h//4, w//2, h//2),
                center=(3*w//8, 3*h//8),
                area=(w//4) * (h//4)
            )
        ]
    
    def detect_obstacles(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect only obstacle-relevant objects.
        
        Args:
            frame: Input image
            
        Returns:
            List of obstacle detections
        """
        all_detections = self.detect(frame)
        return [d for d in all_detections if d.class_id in self.OBSTACLE_CLASSES]
    
    def get_obstacle_map(
        self, 
        frame: np.ndarray,
        grid_size: Tuple[int, int] = (10, 10)
    ) -> np.ndarray:
        """
        Generate an obstacle occupancy grid from detections.
        
        Args:
            frame: Input image
            grid_size: Size of the output grid (rows, cols)
            
        Returns:
            Binary occupancy grid where 1 = obstacle
        """
        h, w = frame.shape[:2]
        grid = np.zeros(grid_size, dtype=np.uint8)
        
        detections = self.detect_obstacles(frame)
        
        cell_h = h // grid_size[0]
        cell_w = w // grid_size[1]
        
        for det in detections:
            cx, cy = det.center
            grid_x = min(cx // cell_w, grid_size[1] - 1)
            grid_y = min(cy // cell_h, grid_size[0] - 1)
            grid[grid_y, grid_x] = 1
        
        return grid
