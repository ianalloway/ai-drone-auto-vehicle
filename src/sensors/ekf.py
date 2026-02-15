"""
Extended Kalman Filter for Drone AI

Implements EKF for state estimation combining multiple sensor inputs.
"""

import numpy as np
from typing import Optional, Tuple
from loguru import logger


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for drone state estimation.
    
    Estimates position, velocity, and orientation by fusing
    GPS, IMU, and other sensor measurements.
    """
    
    def __init__(
        self,
        state_dim: int = 9,
        measurement_dim: int = 6,
        dt: float = 0.01
    ):
        """
        Initialize EKF.
        
        Args:
            state_dim: Dimension of state vector [x, y, z, vx, vy, vz, roll, pitch, yaw]
            measurement_dim: Dimension of measurement vector
            dt: Time step
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.dt = dt
        
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.x = np.zeros(state_dim)
        
        # State covariance
        self.P = np.eye(state_dim) * 1.0
        
        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1
        self.Q[:3, :3] *= 0.01  # Position noise
        self.Q[3:6, 3:6] *= 0.1  # Velocity noise
        self.Q[6:9, 6:9] *= 0.01  # Orientation noise
        
        # Measurement noise covariance
        self.R = np.eye(measurement_dim) * 0.5
        
        # State transition matrix (linearized)
        self.F = np.eye(state_dim)
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt
        
        # Measurement matrix
        self.H = np.zeros((measurement_dim, state_dim))
        self.H[:3, :3] = np.eye(3)  # Position measurements
        self.H[3:6, 6:9] = np.eye(3)  # Orientation measurements
        
        self._initialized = False
        
        logger.info(f"EKF initialized: state_dim={state_dim}, dt={dt}")
    
    def initialize(self, initial_state: np.ndarray):
        """
        Initialize filter with initial state.
        
        Args:
            initial_state: Initial state vector
        """
        self.x = initial_state.copy()
        self._initialized = True
        logger.info("EKF initialized with state")
    
    def predict(self, control_input: Optional[np.ndarray] = None):
        """
        Prediction step of EKF.
        
        Args:
            control_input: Optional control input (acceleration)
        """
        if not self._initialized:
            logger.warning("EKF not initialized, skipping predict")
            return
        
        # State prediction
        self.x = self.F @ self.x
        
        # Apply control input if provided
        if control_input is not None:
            # Assuming control is acceleration [ax, ay, az]
            B = np.zeros((self.state_dim, 3))
            B[3:6, :] = np.eye(3) * self.dt
            self.x += B @ control_input
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray):
        """
        Update step of EKF.
        
        Args:
            measurement: Measurement vector [x, y, z, roll, pitch, yaw]
        """
        if not self._initialized:
            # Initialize with first measurement
            self.x[:3] = measurement[:3]
            if len(measurement) > 3:
                self.x[6:9] = measurement[3:6]
            self._initialized = True
            return
        
        # Innovation
        y = measurement - self.H @ self.x
        
        # Normalize angles
        for i in range(3, len(y)):
            while y[i] > np.pi:
                y[i] -= 2 * np.pi
            while y[i] < -np.pi:
                y[i] += 2 * np.pi
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x.copy()
    
    def get_position(self) -> np.ndarray:
        """Get position estimate [x, y, z]."""
        return self.x[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get velocity estimate [vx, vy, vz]."""
        return self.x[3:6].copy()
    
    def get_orientation(self) -> np.ndarray:
        """Get orientation estimate [roll, pitch, yaw]."""
        return self.x[6:9].copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix."""
        return self.P.copy()
    
    def set_process_noise(self, Q: np.ndarray):
        """Set process noise covariance."""
        self.Q = Q.copy()
    
    def set_measurement_noise(self, R: np.ndarray):
        """Set measurement noise covariance."""
        self.R = R.copy()
