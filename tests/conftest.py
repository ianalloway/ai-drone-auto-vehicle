"""
Shared pytest fixtures for ai-drone-auto-vehicle test suite.

Provides common mocks, test data, and fixtures used across multiple
test modules to avoid duplication and ensure consistent test setup.
"""
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Drone state fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_drone_state():
    """A DroneState representing perfectly nominal operating conditions."""
    from decision.safety import DroneState
    return DroneState(
        battery_percent=80.0,
        altitude=50.0,
        ground_speed=5.0,
        vertical_speed=0.0,
        distance_from_home=100.0,
        signal_strength=90.0,
        gps_satellites=12,
        is_armed=True,
        flight_mode="GUIDED",
        heading=0.0,
        pitch=0.0,
        roll=0.0,
    )


@pytest.fixture
def safety_monitor():
    """A SafetyMonitor with default thresholds."""
    from decision.safety import SafetyMonitor
    return SafetyMonitor()


# ---------------------------------------------------------------------------
# Image / frame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_frame():
    """A plain black 640x480 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """A plain white 640x480 BGR frame."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Sensor data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gps_data():
    """A nominal GPSData reading."""
    from sensors.fusion import GPSData
    return GPSData(
        latitude=37.7749,
        longitude=-122.4194,
        altitude=100.0,
        accuracy=2.0,
        satellites=12,
        timestamp=time.time(),
    )


@pytest.fixture
def imu_data():
    """A nominal IMUData reading (drone hovering level)."""
    from sensors.fusion import IMUData
    return IMUData(
        acceleration=np.array([0.0, 0.0, 9.81]),
        gyroscope=np.array([0.0, 0.0, 0.0]),
        magnetometer=np.array([1.0, 0.0, 0.0]),
        timestamp=time.time(),
    )


@pytest.fixture
def baro_data():
    """A nominal BarometerData reading."""
    from sensors.fusion import BarometerData
    return BarometerData(
        pressure=101325.0,
        temperature=25.0,
        altitude=100.0,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Planning fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """A 10x10 obstacle-free grid for quick A* tests."""
    return np.zeros((10, 10), dtype=np.uint8)


@pytest.fixture
def astar_planner():
    """An AStarPlanner with a 20x20 grid."""
    from planning.astar import AStarPlanner
    return AStarPlanner(grid_size=(20, 20), safety_margin=0)


@pytest.fixture
def rrt_planner_2d():
    """An RRTStarPlanner configured for a 2D 100x100 space."""
    from planning.rrt import RRTStarPlanner
    return RRTStarPlanner(
        bounds=((0.0, 100.0), (0.0, 100.0)),
        step_size=5.0,
        goal_sample_rate=0.2,
        max_iterations=2000,
        search_radius=15.0,
    )


# ---------------------------------------------------------------------------
# MAVLink mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mavlink_controller():
    """A MAVLinkController with no real hardware dependency."""
    from communication.mavlink import MAVLinkController
    ctrl = MAVLinkController(connection_string="udp:127.0.0.1:14550")
    # Manually mark as connected so methods that gate on _connected work
    ctrl._connected = True
    ctrl.connection = None  # No real pymavlink connection
    return ctrl
