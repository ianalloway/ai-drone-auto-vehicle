"""Tests for sensors module: EKF and sensor fusion."""

import time
import numpy as np
import pytest

from src.sensors.ekf import ExtendedKalmanFilter
from src.sensors.fusion import (
    SensorFusion,
    GPSData,
    IMUData,
    BarometerData,
    FusedState,
)


# ─────────────────────────────────────────────────────────────
# ExtendedKalmanFilter
# ─────────────────────────────────────────────────────────────

class TestExtendedKalmanFilter:
    def test_init_state_is_zero(self):
        ekf = ExtendedKalmanFilter()
        assert ekf.x.shape == (9,)
        np.testing.assert_array_equal(ekf.x, np.zeros(9))

    def test_init_not_initialized(self):
        ekf = ExtendedKalmanFilter()
        assert not ekf._initialized

    def test_initialize_sets_state(self):
        ekf = ExtendedKalmanFilter()
        state = np.arange(9, dtype=float)
        ekf.initialize(state)
        np.testing.assert_array_equal(ekf.get_state(), state)
        assert ekf._initialized

    def test_predict_before_init_does_not_crash(self):
        ekf = ExtendedKalmanFilter()
        ekf.predict()  # should just log warning and return

    def test_update_initializes_on_first_call(self):
        ekf = ExtendedKalmanFilter()
        meas = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        ekf.update(meas)
        assert ekf._initialized
        np.testing.assert_array_almost_equal(ekf.get_position(), meas[:3])

    def test_predict_advances_position(self):
        ekf = ExtendedKalmanFilter(dt=1.0)
        state = np.zeros(9)
        state[3:6] = [1.0, 0.0, 0.0]  # vx=1, vy=0, vz=0
        ekf.initialize(state)
        ekf.predict()
        pos = ekf.get_position()
        assert pos[0] == pytest.approx(1.0)

    def test_update_refines_position(self):
        ekf = ExtendedKalmanFilter()
        # Initialise at origin
        ekf.update(np.zeros(6))
        # Feed measurement far from origin – state should move towards it
        meas = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ekf.update(meas)
        pos = ekf.get_position()
        assert pos[0] > 0.0

    def test_get_velocity(self):
        ekf = ExtendedKalmanFilter()
        state = np.zeros(9)
        state[3:6] = [2.0, -1.0, 0.5]
        ekf.initialize(state)
        vel = ekf.get_velocity()
        np.testing.assert_array_almost_equal(vel, [2.0, -1.0, 0.5])

    def test_get_orientation(self):
        ekf = ExtendedKalmanFilter()
        state = np.zeros(9)
        state[6:9] = [0.1, -0.2, 1.57]
        ekf.initialize(state)
        ori = ekf.get_orientation()
        np.testing.assert_array_almost_equal(ori, [0.1, -0.2, 1.57])

    def test_get_covariance_shape(self):
        ekf = ExtendedKalmanFilter()
        cov = ekf.get_covariance()
        assert cov.shape == (9, 9)

    def test_predict_with_control_input(self):
        ekf = ExtendedKalmanFilter(dt=1.0)
        state = np.zeros(9)
        ekf.initialize(state)
        control = np.array([1.0, 0.0, 0.0])
        ekf.predict(control_input=control)
        vel = ekf.get_velocity()
        # velocity should increase by a*dt along x
        assert vel[0] > 0.0

    def test_set_process_noise(self):
        ekf = ExtendedKalmanFilter()
        Q_new = np.eye(9) * 0.5
        ekf.set_process_noise(Q_new)
        np.testing.assert_array_equal(ekf.Q, Q_new)

    def test_set_measurement_noise(self):
        ekf = ExtendedKalmanFilter()
        R_new = np.eye(6) * 0.25
        ekf.set_measurement_noise(R_new)
        np.testing.assert_array_equal(ekf.R, R_new)

    def test_covariance_decreases_after_update(self):
        ekf = ExtendedKalmanFilter()
        initial_trace = np.trace(ekf.get_covariance())
        ekf.update(np.zeros(6))
        ekf.update(np.zeros(6))
        updated_trace = np.trace(ekf.get_covariance())
        assert updated_trace < initial_trace

    def test_state_copy_not_reference(self):
        ekf = ExtendedKalmanFilter()
        ekf.initialize(np.zeros(9))
        state = ekf.get_state()
        state[0] = 999.0
        assert ekf.x[0] != 999.0


# ─────────────────────────────────────────────────────────────
# SensorFusion
# ─────────────────────────────────────────────────────────────

class TestSensorFusion:
    def _make_gps(self, lat=47.6, lon=-122.3, alt=10.0, acc=2.0, sats=10):
        return GPSData(
            latitude=lat, longitude=lon, altitude=alt,
            accuracy=acc, satellites=sats, timestamp=time.time()
        )

    def _make_imu(self):
        return IMUData(
            acceleration=np.array([0.0, 0.0, 9.81]),
            gyroscope=np.zeros(3),
            magnetometer=np.array([1.0, 0.0, 0.0]),
            timestamp=time.time(),
        )

    def _make_baro(self, alt=10.0):
        return BarometerData(
            pressure=101325.0, temperature=20.0,
            altitude=alt, timestamp=time.time()
        )

    def test_init(self):
        sf = SensorFusion()
        assert sf.reference_lat is None
        assert sf.latest_gps is None

    def test_update_gps_sets_reference(self):
        sf = SensorFusion()
        gps = self._make_gps()
        sf.update_gps(gps)
        assert sf.reference_lat == pytest.approx(47.6)
        assert sf.reference_lon == pytest.approx(-122.3)

    def test_update_gps_stores_latest(self):
        sf = SensorFusion()
        gps = self._make_gps()
        sf.update_gps(gps)
        assert sf.latest_gps is gps

    def test_update_imu_stores_latest(self):
        sf = SensorFusion()
        imu = self._make_imu()
        sf.update_imu(imu)
        assert sf.latest_imu is imu

    def test_update_barometer_stores_latest(self):
        sf = SensorFusion()
        baro = self._make_baro()
        sf.update_barometer(baro)
        assert sf.latest_baro is baro

    def test_get_state_returns_fused_state(self):
        sf = SensorFusion()
        state = sf.get_state()
        assert isinstance(state, FusedState)
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.orientation.shape == (3,)

    def test_gps_to_local_origin(self):
        sf = SensorFusion()
        gps = self._make_gps(lat=47.6, lon=-122.3, alt=100.0)
        sf.update_gps(gps)
        local = sf._gps_to_local(47.6, -122.3, 100.0)
        np.testing.assert_array_almost_equal(local, [0.0, 0.0, 0.0], decimal=5)

    def test_gps_to_local_before_reference(self):
        sf = SensorFusion()
        local = sf._gps_to_local(47.6, -122.3, 100.0)
        np.testing.assert_array_equal(local, np.zeros(3))

    def test_orientation_updated_by_imu(self):
        sf = SensorFusion()
        # Use a tilted acceleration vector so roll/pitch will be non-zero
        imu = IMUData(
            acceleration=np.array([1.0, 0.5, 9.81]),  # slight tilt
            gyroscope=np.zeros(3),
            magnetometer=np.array([1.0, 0.0, 0.0]),
            timestamp=time.time(),
        )
        sf.update_imu(imu)
        # Roll and/or pitch should now be non-zero
        assert sf.orientation[0] != 0.0 or sf.orientation[1] != 0.0

    def test_reset_clears_state(self):
        sf = SensorFusion()
        sf.update_gps(self._make_gps())
        sf.update_imu(self._make_imu())
        sf.reset()
        # GPS reference should be cleared
        assert sf.reference_lat is None
        assert sf.reference_lon is None
        assert sf.reference_alt is None
        # Orientation should be zeroed
        np.testing.assert_array_equal(sf.orientation, np.zeros(3))

    def test_gps_availability_flag(self):
        sf = SensorFusion()
        gps = self._make_gps()
        sf.update_gps(gps)
        state = sf.get_state()
        assert state.gps_available is True

    def test_imu_availability_flag_after_recent_update(self):
        sf = SensorFusion()
        imu = self._make_imu()
        sf.update_imu(imu)
        state = sf.get_state()
        # IMU was just updated so it should be available
        assert state.imu_available is True

    def test_position_covariance_shape(self):
        sf = SensorFusion()
        state = sf.get_state()
        assert state.position_covariance.shape == (3, 3)
