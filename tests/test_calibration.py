"""Tests for sensors calibration module."""

import numpy as np
import pytest

from src.sensors.calibration import (
    AccelerometerCalibration,
    AccelerometerCalibrator,
    GyroscopeCalibration,
    GyroscopeCalibrator,
    MagnetometerCalibration,
    MagnetometerCalibrator,
    BarometerCalibration,
    BarometerCalibrator,
    SensorCalibrationManager,
)

GRAVITY = 9.80665


# ─────────────────────────────────────────────────────────────
# AccelerometerCalibration
# ─────────────────────────────────────────────────────────────

class TestAccelerometerCalibration:
    def test_identity_calibration_unchanged(self):
        cal = AccelerometerCalibration()  # offset=0, scale=1, cross_axis=I
        raw = np.array([0.0, 0.0, GRAVITY])
        np.testing.assert_array_almost_equal(cal.apply(raw), raw)

    def test_offset_correction(self):
        cal = AccelerometerCalibration(offset=np.array([0.1, 0.2, 0.3]))
        raw = np.array([0.1, 0.2, 0.3 + GRAVITY])
        result = cal.apply(raw)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, GRAVITY])

    def test_scale_correction(self):
        cal = AccelerometerCalibration(scale=np.array([2.0, 2.0, 2.0]))
        raw = np.array([1.0, 1.0, 1.0])
        result = cal.apply(raw)
        np.testing.assert_array_almost_equal(result, [2.0, 2.0, 2.0])


# ─────────────────────────────────────────────────────────────
# AccelerometerCalibrator
# ─────────────────────────────────────────────────────────────

class _SimulatedAccel:
    """Helper: simulate six-position accel readings with known bias."""
    BIAS = np.array([0.1, -0.2, 0.05])
    SCALE = np.array([1.02, 0.98, 1.01])  # slight scale errors

    @classmethod
    def reading(cls, axis: int, polarity: int) -> np.ndarray:
        """Return reading for axis (0/1/2) and polarity (+1/-1)."""
        r = cls.BIAS.copy()
        r[axis] += polarity * GRAVITY * cls.SCALE[axis]
        return r


def _fill_six_positions(calibrator: AccelerometerCalibrator, n: int = 50):
    """Feed six position samples into the calibrator."""
    rng = np.random.default_rng(0)
    for axis in range(3):
        for polarity in [1, -1]:
            calibrator.start_position()
            reading = _SimulatedAccel.reading(axis, polarity)
            for _ in range(n):
                # Add small noise
                calibrator.add_sample(reading + rng.normal(0, 1e-4, 3))
            calibrator.finish_position()


class TestAccelerometerCalibrator:
    def test_calibrate_recovers_offset(self):
        np.random.seed(42)
        calibrator = AccelerometerCalibrator(samples_per_position=50)
        _fill_six_positions(calibrator)
        cal = calibrator.calibrate()
        np.testing.assert_array_almost_equal(cal.offset, _SimulatedAccel.BIAS, decimal=3)

    def test_calibrate_requires_six_positions(self):
        calibrator = AccelerometerCalibrator()
        calibrator.start_position()
        calibrator.add_sample(np.zeros(3))
        calibrator.finish_position()
        with pytest.raises(ValueError, match="6"):
            calibrator.calibrate()

    def test_finish_position_without_samples_raises(self):
        calibrator = AccelerometerCalibrator()
        calibrator.start_position()
        with pytest.raises(ValueError):
            calibrator.finish_position()

    def test_reset_clears_positions(self):
        calibrator = AccelerometerCalibrator()
        _fill_six_positions(calibrator)
        calibrator.reset()
        with pytest.raises(ValueError):
            calibrator.calibrate()

    def test_apply_calibration_reduces_error(self):
        np.random.seed(0)
        calibrator = AccelerometerCalibrator(samples_per_position=100)
        _fill_six_positions(calibrator, n=100)
        cal = calibrator.calibrate()

        # Verify calibration reduces bias
        raw = _SimulatedAccel.reading(2, 1)          # +Z
        corrected = cal.apply(raw)
        expected = np.array([0.0, 0.0, GRAVITY])

        raw_error = np.linalg.norm(raw - expected)
        cal_error = np.linalg.norm(corrected - expected)
        assert cal_error < raw_error


# ─────────────────────────────────────────────────────────────
# GyroscopeCalibration
# ─────────────────────────────────────────────────────────────

class TestGyroscopeCalibration:
    def test_identity_calibration_unchanged(self):
        cal = GyroscopeCalibration()
        raw = np.array([0.1, -0.2, 0.3])
        np.testing.assert_array_almost_equal(cal.apply(raw), raw)

    def test_bias_subtraction(self):
        bias = np.array([0.01, -0.02, 0.005])
        cal = GyroscopeCalibration(bias=bias)
        raw = np.array([0.11, -0.22, 0.305])
        expected = raw - bias
        np.testing.assert_array_almost_equal(cal.apply(raw), expected)


# ─────────────────────────────────────────────────────────────
# GyroscopeCalibrator
# ─────────────────────────────────────────────────────────────

class TestGyroscopeCalibrator:
    def test_calibrate_estimates_bias(self):
        true_bias = np.array([0.01, -0.02, 0.005])
        calibrator = GyroscopeCalibrator(num_samples=200)
        rng = np.random.default_rng(1)
        for _ in range(200):
            calibrator.add_sample(true_bias + rng.normal(0, 1e-4, 3))
        cal = calibrator.calibrate()
        np.testing.assert_array_almost_equal(cal.bias, true_bias, decimal=3)

    def test_calibrate_no_samples_raises(self):
        calibrator = GyroscopeCalibrator()
        with pytest.raises(ValueError):
            calibrator.calibrate()

    def test_reset_clears_samples(self):
        calibrator = GyroscopeCalibrator()
        calibrator.add_sample(np.zeros(3))
        calibrator.reset()
        with pytest.raises(ValueError):
            calibrator.calibrate()


# ─────────────────────────────────────────────────────────────
# MagnetometerCalibration
# ─────────────────────────────────────────────────────────────

class TestMagnetometerCalibration:
    def test_identity_calibration_unchanged(self):
        cal = MagnetometerCalibration()
        raw = np.array([1.0, 0.5, -0.5])
        np.testing.assert_array_almost_equal(cal.apply(raw), raw)

    def test_hard_iron_removal(self):
        hard = np.array([0.3, -0.2, 0.1])
        cal = MagnetometerCalibration(hard_iron=hard)
        raw = hard + np.array([1.0, 0.0, 0.0])
        result = cal.apply(raw)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])


# ─────────────────────────────────────────────────────────────
# MagnetometerCalibrator
# ─────────────────────────────────────────────────────────────

class TestMagnetometerCalibrator:
    def _sphere_samples(
        self, center: np.ndarray, radius: float, n: int = 100
    ) -> np.ndarray:
        """Generate uniformly distributed points on a sphere with an offset."""
        rng = np.random.default_rng(42)
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = np.arccos(rng.uniform(-1, 1, n))
        pts = radius * np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        return pts + center

    def test_calibrate_removes_hard_iron(self):
        true_hi = np.array([0.4, -0.3, 0.2])
        samples = self._sphere_samples(true_hi, radius=1.0, n=200)
        calibrator = MagnetometerCalibrator()
        for s in samples:
            calibrator.add_sample(s)
        cal = calibrator.calibrate()
        np.testing.assert_array_almost_equal(cal.hard_iron, true_hi, decimal=1)

    def test_calibrate_needs_ten_samples(self):
        calibrator = MagnetometerCalibrator()
        for _ in range(9):
            calibrator.add_sample(np.random.randn(3))
        with pytest.raises(ValueError, match="10"):
            calibrator.calibrate()

    def test_reset_clears_samples(self):
        calibrator = MagnetometerCalibrator()
        for _ in range(20):
            calibrator.add_sample(np.random.randn(3))
        calibrator.reset()
        with pytest.raises(ValueError):
            calibrator.calibrate()


# ─────────────────────────────────────────────────────────────
# BarometerCalibration
# ─────────────────────────────────────────────────────────────

class TestBarometerCalibration:
    def test_identity_calibration(self):
        cal = BarometerCalibration()
        assert cal.apply_pressure(101325.0) == pytest.approx(101325.0)
        assert cal.apply_temperature(20.0) == pytest.approx(20.0)

    def test_pressure_offset(self):
        cal = BarometerCalibration(pressure_offset=100.0)
        assert cal.apply_pressure(101325.0) == pytest.approx(101425.0)

    def test_temperature_offset(self):
        cal = BarometerCalibration(temperature_offset=2.0)
        assert cal.apply_temperature(20.0) == pytest.approx(22.0)

    def test_scale_factor(self):
        cal = BarometerCalibration(scale_factor=1.01)
        assert cal.apply_pressure(100.0) == pytest.approx(101.0)


# ─────────────────────────────────────────────────────────────
# BarometerCalibrator
# ─────────────────────────────────────────────────────────────

class TestBarometerCalibrator:
    SEA_LEVEL_PRESSURE = 101325.0
    KNOWN_ALTITUDE = 0.0

    def test_calibrate_at_sea_level(self):
        calibrator = BarometerCalibrator()
        # Simulate a sensor that reads slightly high
        for _ in range(50):
            calibrator.add_sample(pressure=101200.0, temperature=20.0)
        cal = calibrator.calibrate(
            reference_altitude=self.KNOWN_ALTITUDE,
            reference_pressure=self.SEA_LEVEL_PRESSURE,
        )
        # Offset should correct towards sea-level pressure
        assert cal.pressure_offset == pytest.approx(125.0, abs=1.0)

    def test_calibrate_no_samples_raises(self):
        calibrator = BarometerCalibrator()
        with pytest.raises(ValueError):
            calibrator.calibrate()

    def test_pressure_to_altitude_sea_level(self):
        calibrator = BarometerCalibrator()
        alt = calibrator.pressure_to_altitude(calibrator.SEA_LEVEL_PRESSURE)
        assert alt == pytest.approx(0.0, abs=0.5)

    def test_pressure_to_altitude_decreases_with_pressure(self):
        calibrator = BarometerCalibrator()
        alt_low = calibrator.pressure_to_altitude(90000.0)
        alt_high = calibrator.pressure_to_altitude(95000.0)
        assert alt_low > alt_high

    def test_reset_clears_samples(self):
        calibrator = BarometerCalibrator()
        calibrator.add_sample(101325.0, 20.0)
        calibrator.reset()
        with pytest.raises(ValueError):
            calibrator.calibrate()


# ─────────────────────────────────────────────────────────────
# SensorCalibrationManager
# ─────────────────────────────────────────────────────────────

class TestSensorCalibrationManager:
    def _fully_calibrate(self) -> SensorCalibrationManager:
        manager = SensorCalibrationManager()

        # Gyroscope
        for _ in range(100):
            manager.gyro_calibrator.add_sample(np.zeros(3))
        manager.calibrate_gyroscope()

        # Accelerometer (six positions with default noise)
        _fill_six_positions(manager.accel_calibrator, n=50)
        manager.calibrate_accelerometer()

        # Magnetometer
        for _ in range(50):
            manager.mag_calibrator.add_sample(np.random.randn(3))
        manager.calibrate_magnetometer()

        # Barometer
        for _ in range(20):
            manager.baro_calibrator.add_sample(101325.0, 20.0)
        manager.calibrate_barometer()

        return manager

    def test_not_fully_calibrated_initially(self):
        manager = SensorCalibrationManager()
        assert not manager.is_fully_calibrated()

    def test_fully_calibrated_after_all_sensors(self):
        manager = self._fully_calibrate()
        assert manager.is_fully_calibrated()

    def test_apply_calibrations_returns_four_values(self):
        manager = self._fully_calibrate()
        accel, gyro, mag, pressure = manager.apply_calibrations(
            accel_raw=np.array([0.0, 0.0, GRAVITY]),
            gyro_raw=np.zeros(3),
            mag_raw=np.array([1.0, 0.0, 0.0]),
            pressure_raw=101325.0,
        )
        assert accel.shape == (3,)
        assert gyro.shape == (3,)
        assert mag.shape == (3,)
        assert isinstance(pressure, float)

    def test_apply_calibrations_without_calibration_returns_raw(self):
        manager = SensorCalibrationManager()
        raw_accel = np.array([0.5, 0.5, 9.8])
        accel, _, _, _ = manager.apply_calibrations(
            accel_raw=raw_accel,
            gyro_raw=np.zeros(3),
            mag_raw=np.zeros(3),
            pressure_raw=101325.0,
        )
        np.testing.assert_array_almost_equal(accel, raw_accel)
