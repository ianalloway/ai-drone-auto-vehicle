"""
Sensor Calibration Module for Drone AI

Provides calibration routines for accelerometer, gyroscope, magnetometer,
and barometer sensors used in autonomous drone navigation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class AccelerometerCalibration:
    """Calibration parameters for an accelerometer."""
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    cross_axis: np.ndarray = field(default_factory=lambda: np.eye(3))

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Apply calibration to a raw accelerometer reading."""
        corrected = self.cross_axis @ (raw - self.offset)
        return corrected * self.scale


@dataclass
class GyroscopeCalibration:
    """Calibration parameters for a gyroscope."""
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Apply calibration to a raw gyroscope reading."""
        return (raw - self.bias) * self.scale


@dataclass
class MagnetometerCalibration:
    """Calibration parameters for a magnetometer (hard-iron + soft-iron)."""
    hard_iron: np.ndarray = field(default_factory=lambda: np.zeros(3))
    soft_iron: np.ndarray = field(default_factory=lambda: np.eye(3))

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Apply hard-iron and soft-iron corrections."""
        return self.soft_iron @ (raw - self.hard_iron)


@dataclass
class BarometerCalibration:
    """Calibration parameters for a barometer."""
    pressure_offset: float = 0.0
    temperature_offset: float = 0.0
    scale_factor: float = 1.0

    def apply_pressure(self, raw_pressure: float) -> float:
        """Apply calibration to a raw pressure reading."""
        return (raw_pressure + self.pressure_offset) * self.scale_factor

    def apply_temperature(self, raw_temperature: float) -> float:
        """Apply calibration to a raw temperature reading."""
        return raw_temperature + self.temperature_offset


class AccelerometerCalibrator:
    """
    Calibrates a 3-axis accelerometer using a six-position static method.

    The drone is held stationary in six orientations (±X, ±Y, ±Z facing up)
    and samples are collected.  Offsets and scale factors are then computed
    from the mean readings.
    """

    GRAVITY = 9.80665  # m/s²

    def __init__(self, samples_per_position: int = 100):
        self.samples_per_position = samples_per_position
        self._position_samples: List[List[np.ndarray]] = []
        self._current_samples: List[np.ndarray] = []

    def start_position(self):
        """Begin collecting samples for a new orientation position."""
        self._current_samples = []
        logger.info("AccelerometerCalibrator: collecting samples for new position")

    def add_sample(self, reading: np.ndarray):
        """Add a raw accelerometer sample for the current position."""
        self._current_samples.append(reading.copy())

    def finish_position(self):
        """Finish collecting samples for the current orientation."""
        if not self._current_samples:
            raise ValueError("No samples collected for this position")
        self._position_samples.append(list(self._current_samples))
        self._current_samples = []
        logger.info(
            f"AccelerometerCalibrator: position {len(self._position_samples)} done "
            f"({len(self._position_samples[-1])} samples)"
        )

    def calibrate(self) -> AccelerometerCalibration:
        """
        Compute calibration from collected positions.

        Requires at least 6 positions (one for each axis polarity).

        Returns:
            AccelerometerCalibration with offset and scale.
        """
        if len(self._position_samples) < 6:
            raise ValueError(
                f"Need at least 6 positions, got {len(self._position_samples)}"
            )

        means = np.array([
            np.mean(pos, axis=0) for pos in self._position_samples[:6]
        ])  # shape (6, 3)

        # Pair each axis: assume order is +X, -X, +Y, -Y, +Z, -Z
        offset = np.zeros(3)
        scale = np.ones(3)

        for axis in range(3):
            pos_idx = axis * 2      # +axis
            neg_idx = axis * 2 + 1  # -axis
            pos_mean = means[pos_idx, axis]
            neg_mean = means[neg_idx, axis]

            offset[axis] = (pos_mean + neg_mean) / 2.0
            scale[axis] = (2.0 * self.GRAVITY) / (pos_mean - neg_mean) if (pos_mean - neg_mean) != 0 else 1.0

        cal = AccelerometerCalibration(offset=offset, scale=scale)
        logger.info(f"Accelerometer calibration complete: offset={offset}, scale={scale}")
        return cal

    def reset(self):
        """Clear all collected samples."""
        self._position_samples = []
        self._current_samples = []


class GyroscopeCalibrator:
    """
    Calibrates a gyroscope by averaging readings taken while stationary.

    The estimated bias is subtracted from every subsequent reading.
    """

    def __init__(self, num_samples: int = 200):
        self.num_samples = num_samples
        self._samples: List[np.ndarray] = []

    def add_sample(self, reading: np.ndarray):
        """Add a raw gyroscope sample."""
        self._samples.append(reading.copy())

    def calibrate(self) -> GyroscopeCalibration:
        """
        Compute gyroscope bias from collected stationary samples.

        Returns:
            GyroscopeCalibration with computed bias.
        """
        if not self._samples:
            raise ValueError("No samples collected")

        bias = np.mean(self._samples, axis=0)
        cal = GyroscopeCalibration(bias=bias)
        logger.info(f"Gyroscope calibration complete: bias={bias}")
        return cal

    def reset(self):
        """Clear all collected samples."""
        self._samples = []


class MagnetometerCalibrator:
    """
    Calibrates a magnetometer using ellipsoid fitting.

    Rotating the drone through all orientations produces a cloud of
    magnetometer samples that form an ellipsoid offset from the origin.
    This calibrator fits a sphere to the data to recover the hard-iron
    offset and a scale correction (simplified soft-iron).
    """

    def __init__(self):
        self._samples: List[np.ndarray] = []

    def add_sample(self, reading: np.ndarray):
        """Add a raw magnetometer sample."""
        self._samples.append(reading.copy())

    def calibrate(self) -> MagnetometerCalibration:
        """
        Compute hard-iron offset and simplified soft-iron matrix.

        Uses a min-max approach to estimate hard-iron bias.

        Returns:
            MagnetometerCalibration with hard_iron and soft_iron.
        """
        if len(self._samples) < 10:
            raise ValueError(
                f"Need at least 10 samples, got {len(self._samples)}"
            )

        data = np.array(self._samples)  # (N, 3)

        # Hard-iron: midpoint of the bounding box in each axis
        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        hard_iron = (mins + maxs) / 2.0

        # Simplified soft-iron: equalise the scale across axes
        ranges = (maxs - mins) / 2.0
        avg_range = np.mean(ranges)
        # Avoid division by zero
        scale = np.where(ranges > 0, avg_range / ranges, 1.0)
        soft_iron = np.diag(scale)

        cal = MagnetometerCalibration(hard_iron=hard_iron, soft_iron=soft_iron)
        logger.info(
            f"Magnetometer calibration complete: hard_iron={hard_iron}, scale={scale}"
        )
        return cal

    def reset(self):
        """Clear all collected samples."""
        self._samples = []


class BarometerCalibrator:
    """
    Calibrates a barometer against a known reference pressure.

    The offset is determined from the difference between measured
    pressure and the reference pressure at a known altitude.
    """

    # International standard atmosphere constants
    SEA_LEVEL_PRESSURE = 101325.0  # Pa
    LAPSE_RATE = 0.0065            # K/m
    SEA_LEVEL_TEMP = 288.15        # K
    GRAVITY = 9.80665              # m/s²
    MOLAR_MASS = 0.0289644         # kg/mol
    GAS_CONSTANT = 8.31446         # J/(mol·K)

    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples
        self._pressure_samples: List[float] = []
        self._temperature_samples: List[float] = []

    def add_sample(self, pressure: float, temperature: float):
        """Add a raw barometer sample (pressure in Pa, temperature in °C)."""
        self._pressure_samples.append(pressure)
        self._temperature_samples.append(temperature)

    def calibrate(
        self,
        reference_altitude: float = 0.0,
        reference_pressure: Optional[float] = None,
    ) -> BarometerCalibration:
        """
        Compute pressure offset relative to a reference altitude.

        Args:
            reference_altitude: Known altitude at which samples were taken (m).
            reference_pressure: Known reference pressure (Pa).  If None, the
                                 ISA pressure at reference_altitude is used.

        Returns:
            BarometerCalibration with pressure_offset and temperature_offset.
        """
        if not self._pressure_samples:
            raise ValueError("No samples collected")

        measured_pressure = float(np.mean(self._pressure_samples))
        measured_temperature = float(np.mean(self._temperature_samples))

        if reference_pressure is None:
            # ISA pressure at the given altitude
            reference_pressure = self.SEA_LEVEL_PRESSURE * (
                1.0 - self.LAPSE_RATE * reference_altitude / self.SEA_LEVEL_TEMP
            ) ** (self.GRAVITY * self.MOLAR_MASS / (self.GAS_CONSTANT * self.LAPSE_RATE))

        pressure_offset = reference_pressure - measured_pressure

        # Standard temperature at the reference altitude
        reference_temperature_c = (
            self.SEA_LEVEL_TEMP - self.LAPSE_RATE * reference_altitude - 273.15
        )
        temperature_offset = reference_temperature_c - measured_temperature

        cal = BarometerCalibration(
            pressure_offset=pressure_offset,
            temperature_offset=temperature_offset,
        )
        logger.info(
            f"Barometer calibration complete: pressure_offset={pressure_offset:.2f} Pa, "
            f"temperature_offset={temperature_offset:.2f} °C"
        )
        return cal

    def pressure_to_altitude(
        self,
        pressure: float,
        reference_pressure: Optional[float] = None,
    ) -> float:
        """
        Convert pressure to altitude using the barometric formula.

        Args:
            pressure: Absolute pressure in Pa.
            reference_pressure: Reference pressure at sea level (Pa).
                                 Defaults to ISA sea-level pressure.

        Returns:
            Estimated altitude above the reference level in metres.
        """
        p0 = reference_pressure if reference_pressure is not None else self.SEA_LEVEL_PRESSURE
        altitude = (
            self.SEA_LEVEL_TEMP / self.LAPSE_RATE
            * (1.0 - (pressure / p0) ** (self.GAS_CONSTANT * self.LAPSE_RATE / (self.GRAVITY * self.MOLAR_MASS)))
        )
        return altitude

    def reset(self):
        """Clear all collected samples."""
        self._pressure_samples = []
        self._temperature_samples = []


class SensorCalibrationManager:
    """
    Top-level manager that coordinates calibration of all drone sensors.

    Typical usage::

        manager = SensorCalibrationManager()
        # --- Gyroscope (drone stationary) ---
        for reading in gyro_stream:
            manager.gyro_calibrator.add_sample(reading)
        gyro_cal = manager.calibrate_gyroscope()

        # --- Accelerometer (six positions) ---
        for orientation_idx in range(6):
            manager.accel_calibrator.start_position()
            for reading in accel_stream:
                manager.accel_calibrator.add_sample(reading)
            manager.accel_calibrator.finish_position()
        accel_cal = manager.calibrate_accelerometer()
    """

    def __init__(
        self,
        accel_samples_per_position: int = 100,
        gyro_samples: int = 200,
        baro_samples: int = 50,
    ):
        self.accel_calibrator = AccelerometerCalibrator(
            samples_per_position=accel_samples_per_position
        )
        self.gyro_calibrator = GyroscopeCalibrator(num_samples=gyro_samples)
        self.mag_calibrator = MagnetometerCalibrator()
        self.baro_calibrator = BarometerCalibrator(num_samples=baro_samples)

        self.accel_cal: Optional[AccelerometerCalibration] = None
        self.gyro_cal: Optional[GyroscopeCalibration] = None
        self.mag_cal: Optional[MagnetometerCalibration] = None
        self.baro_cal: Optional[BarometerCalibration] = None

        logger.info("SensorCalibrationManager initialized")

    def calibrate_accelerometer(self) -> AccelerometerCalibration:
        """Run accelerometer calibration and store result."""
        self.accel_cal = self.accel_calibrator.calibrate()
        return self.accel_cal

    def calibrate_gyroscope(self) -> GyroscopeCalibration:
        """Run gyroscope calibration and store result."""
        self.gyro_cal = self.gyro_calibrator.calibrate()
        return self.gyro_cal

    def calibrate_magnetometer(self) -> MagnetometerCalibration:
        """Run magnetometer calibration and store result."""
        self.mag_cal = self.mag_calibrator.calibrate()
        return self.mag_cal

    def calibrate_barometer(
        self,
        reference_altitude: float = 0.0,
        reference_pressure: Optional[float] = None,
    ) -> BarometerCalibration:
        """Run barometer calibration and store result."""
        self.baro_cal = self.baro_calibrator.calibrate(
            reference_altitude=reference_altitude,
            reference_pressure=reference_pressure,
        )
        return self.baro_cal

    def is_fully_calibrated(self) -> bool:
        """Return True only if all four sensors have been calibrated."""
        return all(
            c is not None
            for c in (self.accel_cal, self.gyro_cal, self.mag_cal, self.baro_cal)
        )

    def apply_calibrations(
        self,
        accel_raw: np.ndarray,
        gyro_raw: np.ndarray,
        mag_raw: np.ndarray,
        pressure_raw: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Apply all stored calibrations to raw sensor readings.

        Returns calibrated (accel, gyro, mag, pressure) values.
        If a calibration is not yet available for a sensor, the raw
        value is returned unchanged.
        """
        accel = self.accel_cal.apply(accel_raw) if self.accel_cal else accel_raw.copy()
        gyro = self.gyro_cal.apply(gyro_raw) if self.gyro_cal else gyro_raw.copy()
        mag = self.mag_cal.apply(mag_raw) if self.mag_cal else mag_raw.copy()
        pressure = (
            self.baro_cal.apply_pressure(pressure_raw)
            if self.baro_cal
            else pressure_raw
        )
        return accel, gyro, mag, pressure
