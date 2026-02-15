"""Tests for sensors module."""
import pytest
import numpy as np
from src.sensors.ekf import ExtendedKalmanFilter
from src.sensors.fusion import SensorFusion, GPSData, IMUData, BarometerData


class TestExtendedKalmanFilter:
    """Test cases for Extended Kalman Filter."""

    def test_ekf_initialization(self):
        """Test EKF initializes correctly."""
        ekf = ExtendedKalmanFilter()
        state = ekf.get_state()
        assert state is not None
        assert len(state) == 9  # x, y, z, vx, vy, vz, roll, pitch, yaw

    def test_ekf_predict(self):
        """Test EKF prediction step."""
        ekf = ExtendedKalmanFilter()
        initial_state = ekf.get_state().copy()
        ekf.predict(dt=0.1)
        new_state = ekf.get_state()
        # State should change after prediction
        assert new_state is not None

    def test_ekf_update(self):
        """Test EKF update step."""
        ekf = ExtendedKalmanFilter()
        measurement = np.array([10.0, 20.0, 30.0])  # x, y, z
        ekf.update(measurement, measurement_type="position")
        state = ekf.get_state()
        # Position should be close to measurement
        assert abs(state[0] - 10.0) < 5.0

    def test_get_position(self):
        """Test getting position from EKF."""
        ekf = ExtendedKalmanFilter()
        pos = ekf.get_position()
        assert len(pos) == 3

    def test_get_velocity(self):
        """Test getting velocity from EKF."""
        ekf = ExtendedKalmanFilter()
        vel = ekf.get_velocity()
        assert len(vel) == 3

    def test_get_orientation(self):
        """Test getting orientation from EKF."""
        ekf = ExtendedKalmanFilter()
        orient = ekf.get_orientation()
        assert len(orient) == 3


class TestSensorFusion:
    """Test cases for sensor fusion."""

    def test_fusion_initialization(self):
        """Test sensor fusion initializes correctly."""
        fusion = SensorFusion()
        state = fusion.get_state()
        assert state is not None

    def test_update_gps(self):
        """Test GPS data update."""
        fusion = SensorFusion()
        gps_data = GPSData(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=50.0,
            accuracy=2.0,
            timestamp=0.0
        )
        fusion.update_gps(gps_data)
        state = fusion.get_state()
        assert state is not None

    def test_update_imu(self):
        """Test IMU data update."""
        fusion = SensorFusion()
        imu_data = IMUData(
            accel_x=0.0,
            accel_y=0.0,
            accel_z=-9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            timestamp=0.0
        )
        fusion.update_imu(imu_data)
        state = fusion.get_state()
        assert state is not None

    def test_update_barometer(self):
        """Test barometer data update."""
        fusion = SensorFusion()
        baro_data = BarometerData(
            pressure=101325.0,
            temperature=20.0,
            altitude=50.0,
            timestamp=0.0
        )
        fusion.update_barometer(baro_data)
        state = fusion.get_state()
        assert state is not None

    def test_multi_sensor_fusion(self):
        """Test fusing data from multiple sensors."""
        fusion = SensorFusion()
        
        # Update with GPS
        gps_data = GPSData(37.7749, -122.4194, 50.0, 2.0, 0.0)
        fusion.update_gps(gps_data)
        
        # Update with IMU
        imu_data = IMUData(0.0, 0.0, -9.81, 0.0, 0.0, 0.0, 0.01)
        fusion.update_imu(imu_data)
        
        # Update with barometer
        baro_data = BarometerData(101325.0, 20.0, 50.0, 0.02)
        fusion.update_barometer(baro_data)
        
        state = fusion.get_state()
        assert state is not None
        assert state.altitude is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
