"""
Sensor Fusion Module for Drone AI

Combines data from multiple sensors for robust state estimation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
from loguru import logger
import time

from .ekf import ExtendedKalmanFilter


@dataclass
class GPSData:
    """GPS measurement data."""
    latitude: float
    longitude: float
    altitude: float
    accuracy: float
    satellites: int
    timestamp: float


@dataclass
class IMUData:
    """IMU measurement data."""
    acceleration: np.ndarray  # [ax, ay, az]
    gyroscope: np.ndarray     # [gx, gy, gz]
    magnetometer: np.ndarray  # [mx, my, mz]
    timestamp: float


@dataclass
class BarometerData:
    """Barometer measurement data."""
    pressure: float
    temperature: float
    altitude: float
    timestamp: float


@dataclass
class FusedState:
    """Fused state estimate."""
    position: np.ndarray      # [x, y, z] in local frame
    velocity: np.ndarray      # [vx, vy, vz]
    orientation: np.ndarray   # [roll, pitch, yaw]
    position_covariance: np.ndarray
    timestamp: float
    gps_available: bool
    imu_available: bool


class SensorFusion:
    """
    Multi-sensor fusion for drone state estimation.
    
    Combines GPS, IMU, barometer, and other sensors using
    Extended Kalman Filter for robust state estimation.
    """
    
    def __init__(
        self,
        update_rate: float = 100.0,
        gps_weight: float = 1.0,
        baro_weight: float = 0.8
    ):
        """
        Initialize sensor fusion.
        
        Args:
            update_rate: Fusion update rate in Hz
            gps_weight: Weight for GPS measurements
            baro_weight: Weight for barometer altitude
        """
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        self.gps_weight = gps_weight
        self.baro_weight = baro_weight
        
        # Extended Kalman Filter
        self.ekf = ExtendedKalmanFilter(state_dim=9, measurement_dim=6, dt=self.dt)
        
        # Reference point for local coordinates
        self.reference_lat: Optional[float] = None
        self.reference_lon: Optional[float] = None
        self.reference_alt: Optional[float] = None
        
        # Latest sensor data
        self.latest_gps: Optional[GPSData] = None
        self.latest_imu: Optional[IMUData] = None
        self.latest_baro: Optional[BarometerData] = None
        
        # Sensor health
        self.gps_timeout = 2.0  # seconds
        self.imu_timeout = 0.1
        
        # Complementary filter for orientation
        self.orientation = np.zeros(3)
        self.alpha = 0.98  # Complementary filter coefficient
        
        logger.info("SensorFusion initialized")
    
    def update_gps(self, data: GPSData):
        """
        Update with new GPS data.
        
        Args:
            data: GPS measurement
        """
        self.latest_gps = data
        
        # Set reference point on first GPS fix
        if self.reference_lat is None:
            self.reference_lat = data.latitude
            self.reference_lon = data.longitude
            self.reference_alt = data.altitude
            logger.info(f"GPS reference set: {data.latitude}, {data.longitude}")
        
        # Convert to local coordinates
        local_pos = self._gps_to_local(data.latitude, data.longitude, data.altitude)
        
        # Create measurement vector
        measurement = np.zeros(6)
        measurement[:3] = local_pos
        measurement[3:6] = self.orientation
        
        # Adjust measurement noise based on GPS accuracy
        R = self.ekf.R.copy()
        R[:3, :3] *= (data.accuracy / 2.0) ** 2
        self.ekf.set_measurement_noise(R)
        
        # Update EKF
        self.ekf.update(measurement)
    
    def update_imu(self, data: IMUData):
        """
        Update with new IMU data.
        
        Args:
            data: IMU measurement
        """
        self.latest_imu = data
        
        # Update orientation using complementary filter
        self._update_orientation(data)
        
        # Use acceleration as control input for EKF prediction
        self.ekf.predict(data.acceleration)
    
    def update_barometer(self, data: BarometerData):
        """
        Update with new barometer data.
        
        Args:
            data: Barometer measurement
        """
        self.latest_baro = data
        
        # Fuse barometer altitude with GPS altitude
        if self.reference_alt is not None:
            baro_alt = data.altitude - self.reference_alt
            
            # Get current state
            state = self.ekf.get_state()
            
            # Weighted average of GPS and baro altitude
            if self.latest_gps is not None:
                gps_age = time.time() - self.latest_gps.timestamp
                if gps_age < self.gps_timeout:
                    state[2] = (self.gps_weight * state[2] + 
                               self.baro_weight * baro_alt) / (self.gps_weight + self.baro_weight)
    
    def _update_orientation(self, imu: IMUData):
        """Update orientation using complementary filter."""
        # Calculate orientation from accelerometer
        ax, ay, az = imu.acceleration
        accel_roll = np.arctan2(ay, az)
        accel_pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        
        # Integrate gyroscope
        gx, gy, gz = imu.gyroscope
        gyro_roll = self.orientation[0] + gx * self.dt
        gyro_pitch = self.orientation[1] + gy * self.dt
        gyro_yaw = self.orientation[2] + gz * self.dt
        
        # Complementary filter
        self.orientation[0] = self.alpha * gyro_roll + (1 - self.alpha) * accel_roll
        self.orientation[1] = self.alpha * gyro_pitch + (1 - self.alpha) * accel_pitch
        self.orientation[2] = gyro_yaw  # Yaw from gyro only (or use magnetometer)
        
        # Use magnetometer for yaw if available
        if np.linalg.norm(imu.magnetometer) > 0:
            mx, my, mz = imu.magnetometer
            mag_yaw = np.arctan2(my, mx)
            self.orientation[2] = self.alpha * gyro_yaw + (1 - self.alpha) * mag_yaw
    
    def _gps_to_local(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Convert GPS coordinates to local frame."""
        if self.reference_lat is None:
            return np.zeros(3)
        
        # Simple flat-earth approximation (valid for small distances)
        R_earth = 6371000  # meters
        
        dlat = np.radians(lat - self.reference_lat)
        dlon = np.radians(lon - self.reference_lon)
        
        x = R_earth * dlon * np.cos(np.radians(self.reference_lat))
        y = R_earth * dlat
        z = alt - self.reference_alt
        
        return np.array([x, y, z])
    
    def get_state(self) -> FusedState:
        """
        Get current fused state estimate.
        
        Returns:
            FusedState with position, velocity, orientation
        """
        now = time.time()
        
        # Check sensor availability
        gps_available = (self.latest_gps is not None and 
                        now - self.latest_gps.timestamp < self.gps_timeout)
        imu_available = (self.latest_imu is not None and 
                        now - self.latest_imu.timestamp < self.imu_timeout)
        
        return FusedState(
            position=self.ekf.get_position(),
            velocity=self.ekf.get_velocity(),
            orientation=self.orientation.copy(),
            position_covariance=self.ekf.get_covariance()[:3, :3],
            timestamp=now,
            gps_available=gps_available,
            imu_available=imu_available
        )
    
    def reset(self):
        """Reset sensor fusion state."""
        self.ekf = ExtendedKalmanFilter(state_dim=9, measurement_dim=6, dt=self.dt)
        self.reference_lat = None
        self.reference_lon = None
        self.reference_alt = None
        self.orientation = np.zeros(3)
        logger.info("SensorFusion reset")
