"""Tests for communication module: MAVLink controller."""

import pytest

from src.communication.mavlink import MAVLinkController, FlightMode, Telemetry


# ─────────────────────────────────────────────────────────────
# MAVLinkController
# ─────────────────────────────────────────────────────────────

class TestMAVLinkController:
    def test_init_defaults(self):
        ctrl = MAVLinkController()
        assert ctrl.connection_string == "udp:127.0.0.1:14550"
        assert ctrl._connected is False
        assert ctrl._armed is False

    def test_init_custom_string(self):
        ctrl = MAVLinkController(connection_string="tcp:192.168.1.1:5760")
        assert ctrl.connection_string == "tcp:192.168.1.1:5760"

    def test_connect_without_pymavlink(self):
        """connect() should succeed via mock path when pymavlink is absent."""
        ctrl = MAVLinkController()
        result = ctrl.connect(timeout=1.0)
        assert result is True
        assert ctrl.is_connected()

    def test_disconnect(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        ctrl.disconnect()
        assert ctrl._connected is False

    def test_arm_when_connected(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        result = ctrl.arm()
        assert result is True
        assert ctrl.is_armed()

    def test_arm_when_not_connected(self):
        ctrl = MAVLinkController()
        result = ctrl.arm()
        assert result is False
        assert not ctrl.is_armed()

    def test_disarm_when_connected(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        ctrl.arm()
        result = ctrl.disarm()
        assert result is True
        assert not ctrl.is_armed()

    def test_disarm_when_not_connected(self):
        ctrl = MAVLinkController()
        result = ctrl.disarm()
        assert result is False

    def test_takeoff_when_armed(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        ctrl.arm()
        result = ctrl.takeoff(10.0)
        assert result is True

    def test_takeoff_when_not_armed(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        result = ctrl.takeoff(10.0)
        assert result is False

    def test_land(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        result = ctrl.land()
        assert result is True

    def test_goto(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        result = ctrl.goto(47.6, -122.3, 30.0)
        assert result is True

    def test_set_velocity(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        result = ctrl.set_velocity(1.0, 0.0, 0.0)
        assert result is True

    def test_set_mode(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        result = ctrl.set_mode(FlightMode.GUIDED)
        assert result is True

    def test_return_to_launch(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        result = ctrl.return_to_launch()
        assert result is True

    def test_get_telemetry_when_connected(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        telemetry = ctrl.get_telemetry()
        assert telemetry is not None
        assert isinstance(telemetry, Telemetry)

    def test_get_telemetry_when_not_connected(self):
        ctrl = MAVLinkController()
        assert ctrl.get_telemetry() is None

    def test_mock_telemetry_fields(self):
        ctrl = MAVLinkController()
        ctrl.connect(timeout=1.0)
        t = ctrl.get_telemetry()
        assert isinstance(t.latitude, float)
        assert isinstance(t.longitude, float)
        assert isinstance(t.battery_percent, float)
        assert isinstance(t.armed, bool)
        assert t.gps_satellites > 0

    def test_register_telemetry_callback(self):
        ctrl = MAVLinkController()
        received = []
        ctrl.register_telemetry_callback(received.append)
        assert len(ctrl.telemetry_callbacks) == 1

    def test_flight_modes_enum(self):
        modes = list(FlightMode)
        assert FlightMode.GUIDED in modes
        assert FlightMode.RTL in modes
        assert FlightMode.LAND in modes


# ─────────────────────────────────────────────────────────────
# Telemetry dataclass
# ─────────────────────────────────────────────────────────────

class TestTelemetry:
    def test_telemetry_fields(self):
        import time
        t = Telemetry(
            latitude=47.6,
            longitude=-122.3,
            altitude=50.0,
            relative_altitude=50.0,
            heading=90.0,
            ground_speed=5.0,
            vertical_speed=0.5,
            battery_voltage=12.4,
            battery_percent=80.0,
            armed=True,
            flight_mode="GUIDED",
            gps_satellites=12,
            roll=0.0,
            pitch=2.0,
            yaw=1.57,
            timestamp=time.time(),
        )
        assert t.latitude == pytest.approx(47.6)
        assert t.flight_mode == "GUIDED"
        assert t.armed is True
