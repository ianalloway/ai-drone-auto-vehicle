"""
Unit tests for the SafetyMonitor module.

These tests verify correct safety response behavior, and also document two
bugs in the current code that PR #8 is designed to fix.

Bug 1 — SafetyLevel is not orderable (CRASH)
---------------------------------------------
SafetyLevel uses plain `Enum` with integer values, but Python Enum members
do not support `>` / `<` comparisons unless `IntEnum` is used.
check_state() calls `max(a.level for a in self.alerts)` which raises:
    TypeError: '>' not supported between instances of 'SafetyLevel' and 'SafetyLevel'
whenever MORE THAN ONE alert fires simultaneously.  This is a crash, not
merely a wrong answer.

Bug 2 — SafetyAction uses string values (WRONG PRIORITY ORDER)
--------------------------------------------------------------
SafetyAction uses string enum values, so `.value` comparisons are
lexicographic rather than severity-ordered:
    CONTINUE       = "continue"      -> lex rank 1  (intended rank 1)
    SLOW_DOWN      = "slow_down"     -> lex rank 6  (intended rank 2)  ← too high
    HOVER          = "hover"         -> lex rank 3  (intended rank 3)
    RETURN_HOME    = "return_home"   -> lex rank 5  (intended rank 4)
    EMERGENCY_LAND = "emergency_land"-> lex rank 2  (intended rank 5)  ← too low!
    KILL_MOTORS    = "kill_motors"   -> lex rank 4  (intended rank 6)

Because Bug 1 crashes first, Bug 2 is currently masked for multi-alert
scenarios, but it would manifest once Bug 1 is fixed in isolation.

PR #8 fixes both bugs by switching SafetyAction to integer values and
SafetyLevel to IntEnum (or equivalent orderable type).
"""

import pytest
import time
from unittest.mock import MagicMock
from decision.safety import (
    SafetyMonitor,
    SafetyAction,
    SafetyLevel,
    SafetyAlert,
    DroneState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides) -> DroneState:
    """Return a safe nominal DroneState, with any field overridden."""
    defaults = dict(
        battery_percent=80.0,
        altitude=50.0,
        ground_speed=5.0,
        vertical_speed=1.0,
        distance_from_home=100.0,
        signal_strength=80.0,
        gps_satellites=10,
        is_armed=True,
        flight_mode="AUTO",
        heading=0.0,
        pitch=0.0,
        roll=0.0,
    )
    defaults.update(overrides)
    return DroneState(**defaults)


def make_monitor(**overrides) -> SafetyMonitor:
    """Return a SafetyMonitor with default thresholds, optionally overridden."""
    defaults = dict(
        battery_warning=30.0,
        battery_critical=15.0,
        battery_emergency=10.0,
        max_altitude=120.0,
        max_distance=500.0,
        max_speed=20.0,
        min_satellites=6,
        signal_warning=40.0,
        signal_critical=20.0,
        max_tilt=45.0,
    )
    defaults.update(overrides)
    return SafetyMonitor(**defaults)


# ---------------------------------------------------------------------------
# Basic safety check tests (nominal / no-alert paths)
# ---------------------------------------------------------------------------

class TestNominalState:
    """All checks pass for a drone in nominal state."""

    def test_nominal_state_returns_continue(self):
        monitor = make_monitor()
        state = make_state()
        assert monitor.check_state(state) == SafetyAction.CONTINUE

    def test_nominal_state_no_alerts(self):
        monitor = make_monitor()
        state = make_state()
        monitor.check_state(state)
        assert monitor.alerts == []

    def test_nominal_state_level_is_normal(self):
        monitor = make_monitor()
        state = make_state()
        monitor.check_state(state)
        assert monitor.current_level == SafetyLevel.NORMAL


# ---------------------------------------------------------------------------
# Battery checks
# ---------------------------------------------------------------------------

class TestBatteryChecks:
    """Battery threshold behaviour."""

    def test_battery_warning_returns_continue_with_alert(self):
        monitor = make_monitor()
        # 25% is between battery_warning (30) and battery_critical (15)
        state = make_state(battery_percent=25.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].level == SafetyLevel.WARNING

    def test_battery_critical_returns_return_home(self):
        monitor = make_monitor()
        state = make_state(battery_percent=12.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME

    def test_battery_critical_alert_level_is_critical(self):
        monitor = make_monitor()
        state = make_state(battery_percent=12.0)
        monitor.check_state(state)
        battery_alerts = [a for a in monitor.alerts if a.source == "battery"]
        assert battery_alerts[0].level == SafetyLevel.CRITICAL

    def test_battery_emergency_returns_emergency_land(self):
        """Battery at or below emergency threshold must trigger EMERGENCY_LAND."""
        monitor = make_monitor()
        state = make_state(battery_percent=8.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.EMERGENCY_LAND

    def test_battery_exactly_at_emergency_threshold(self):
        monitor = make_monitor(battery_emergency=10.0)
        state = make_state(battery_percent=10.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.EMERGENCY_LAND

    def test_battery_above_emergency_threshold_not_emergency(self):
        monitor = make_monitor(battery_emergency=10.0)
        state = make_state(battery_percent=10.1)
        action = monitor.check_state(state)
        assert action != SafetyAction.EMERGENCY_LAND

    def test_battery_emergency_alert_level_is_emergency(self):
        monitor = make_monitor()
        state = make_state(battery_percent=5.0)
        monitor.check_state(state)
        battery_alerts = [a for a in monitor.alerts if a.source == "battery"]
        assert battery_alerts[0].level == SafetyLevel.EMERGENCY


# ---------------------------------------------------------------------------
# Altitude checks
# ---------------------------------------------------------------------------

class TestAltitudeChecks:

    def test_altitude_within_limit_no_alert(self):
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=119.9)
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE
        assert not any(a.source == "altitude" for a in monitor.alerts)

    def test_altitude_exceeded_returns_slow_down(self):
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=121.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.SLOW_DOWN

    def test_altitude_exceeded_has_warning_alert(self):
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=150.0)
        monitor.check_state(state)
        alt_alerts = [a for a in monitor.alerts if a.source == "altitude"]
        assert alt_alerts
        assert alt_alerts[0].level == SafetyLevel.WARNING


# ---------------------------------------------------------------------------
# Distance / geofence checks
# ---------------------------------------------------------------------------

class TestDistanceChecks:

    def test_within_geofence_no_alert(self):
        monitor = make_monitor(max_distance=500.0)
        state = make_state(distance_from_home=400.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE

    def test_approaching_max_distance_slows_down(self):
        """At 90 %+ of max_distance, the drone should slow down."""
        monitor = make_monitor(max_distance=500.0)
        state = make_state(distance_from_home=460.0)  # 92 %
        action = monitor.check_state(state)
        assert action == SafetyAction.SLOW_DOWN

    def test_geofence_breach_returns_return_home(self):
        monitor = make_monitor(max_distance=500.0)
        state = make_state(distance_from_home=510.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME

    def test_geofence_breach_alert_is_critical(self):
        monitor = make_monitor(max_distance=500.0)
        state = make_state(distance_from_home=600.0)
        monitor.check_state(state)
        gf_alerts = [a for a in monitor.alerts if a.source == "geofence"]
        assert gf_alerts
        assert gf_alerts[0].level == SafetyLevel.CRITICAL

    def test_geofence_disabled_no_geofence_alert(self):
        monitor = make_monitor(max_distance=500.0)
        monitor.disable_geofence()
        state = make_state(distance_from_home=1000.0)
        action = monitor.check_state(state)
        # geofence alert gone; may still get SLOW_DOWN from 90 % check
        assert not any(a.source == "geofence" for a in monitor.alerts)

    def test_set_geofence_updates_radius(self):
        monitor = make_monitor(max_distance=500.0)
        monitor.set_geofence(center=(0.0, 0.0), radius=200.0)
        state = make_state(distance_from_home=250.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME


# ---------------------------------------------------------------------------
# Speed checks
# ---------------------------------------------------------------------------

class TestSpeedChecks:

    def test_within_speed_limit_no_alert(self):
        monitor = make_monitor(max_speed=20.0)
        state = make_state(ground_speed=19.9)
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE

    def test_speed_exceeded_returns_slow_down(self):
        monitor = make_monitor(max_speed=20.0)
        state = make_state(ground_speed=25.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.SLOW_DOWN


# ---------------------------------------------------------------------------
# GPS checks
# ---------------------------------------------------------------------------

class TestGpsChecks:

    def test_sufficient_satellites_no_alert(self):
        monitor = make_monitor(min_satellites=6)
        state = make_state(gps_satellites=8)
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE

    def test_low_satellites_returns_hover(self):
        monitor = make_monitor(min_satellites=6)
        state = make_state(gps_satellites=4)
        action = monitor.check_state(state)
        assert action == SafetyAction.HOVER

    def test_low_satellites_alert_level_is_warning(self):
        monitor = make_monitor(min_satellites=6)
        state = make_state(gps_satellites=3)
        monitor.check_state(state)
        gps_alerts = [a for a in monitor.alerts if a.source == "gps"]
        assert gps_alerts
        assert gps_alerts[0].level == SafetyLevel.WARNING


# ---------------------------------------------------------------------------
# Signal checks
# ---------------------------------------------------------------------------

class TestSignalChecks:

    def test_good_signal_no_alert(self):
        monitor = make_monitor(signal_warning=40.0, signal_critical=20.0)
        state = make_state(signal_strength=80.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE

    def test_weak_signal_returns_slow_down(self):
        monitor = make_monitor(signal_warning=40.0, signal_critical=20.0)
        state = make_state(signal_strength=30.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.SLOW_DOWN

    def test_critical_signal_returns_return_home(self):
        monitor = make_monitor(signal_warning=40.0, signal_critical=20.0)
        state = make_state(signal_strength=15.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME


# ---------------------------------------------------------------------------
# Attitude checks
# ---------------------------------------------------------------------------

class TestAttitudeChecks:

    def test_normal_attitude_no_alert(self):
        monitor = make_monitor(max_tilt=45.0)
        state = make_state(pitch=5.0, roll=10.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE

    def test_excessive_pitch_returns_hover(self):
        monitor = make_monitor(max_tilt=45.0)
        state = make_state(pitch=50.0, roll=0.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.HOVER

    def test_excessive_roll_returns_hover(self):
        monitor = make_monitor(max_tilt=45.0)
        state = make_state(pitch=0.0, roll=50.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.HOVER

    def test_attitude_alert_level_is_critical(self):
        monitor = make_monitor(max_tilt=45.0)
        state = make_state(pitch=60.0, roll=0.0)
        monitor.check_state(state)
        att_alerts = [a for a in monitor.alerts if a.source == "attitude"]
        assert att_alerts
        assert att_alerts[0].level == SafetyLevel.CRITICAL


# ---------------------------------------------------------------------------
# current_level reflects the worst active alert
# ---------------------------------------------------------------------------

class TestCurrentLevel:
    """Verify that current_level is set to the max SafetyLevel among all alerts."""

    def test_current_level_normal_when_no_alerts(self):
        monitor = make_monitor()
        state = make_state()
        monitor.check_state(state)
        assert monitor.current_level == SafetyLevel.NORMAL

    def test_current_level_warning_for_altitude_breach(self):
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=150.0)
        monitor.check_state(state)
        assert monitor.current_level == SafetyLevel.WARNING

    def test_current_level_critical_for_geofence_breach(self):
        monitor = make_monitor(max_distance=500.0)
        state = make_state(distance_from_home=600.0)
        monitor.check_state(state)
        assert monitor.current_level == SafetyLevel.CRITICAL

    def test_current_level_emergency_for_battery_emergency(self):
        monitor = make_monitor()
        state = make_state(battery_percent=5.0)
        monitor.check_state(state)
        assert monitor.current_level == SafetyLevel.EMERGENCY

    def test_current_level_reflects_worst_of_multiple_alerts(self):
        """When altitude (WARNING) and geofence (CRITICAL) both fire, level is CRITICAL."""
        monitor = make_monitor(max_altitude=120.0, max_distance=500.0)
        state = make_state(altitude=150.0, distance_from_home=600.0)
        monitor.check_state(state)
        assert monitor.current_level == SafetyLevel.CRITICAL

    def test_current_level_clears_between_calls(self):
        """current_level must reset to NORMAL once the alert condition is resolved."""
        monitor = make_monitor(max_altitude=120.0)
        bad_state = make_state(altitude=150.0)
        monitor.check_state(bad_state)
        assert monitor.current_level == SafetyLevel.WARNING

        good_state = make_state(altitude=50.0)
        monitor.check_state(good_state)
        assert monitor.current_level == SafetyLevel.NORMAL


# ---------------------------------------------------------------------------
# Highest-severity action selection — tests that expose the string-value bug
# ---------------------------------------------------------------------------

class TestHighestSeverityActionSelection:
    """
    These tests verify that check_state always returns the highest-severity
    action when multiple conditions fire simultaneously.

    KNOWN BUGS (pre PR #8):
      • Bug 1 (crash): SafetyLevel is not orderable → TypeError in max() when
        2+ alerts fire.  All multi-alert tests fail with TypeError until fixed.
      • Bug 2 (wrong result): SafetyAction string values are not severity-ordered.
        Once Bug 1 is fixed, tests like battery_emergency_beats_altitude will
        still fail because "slow_down" > "emergency_land" lexicographically.

    All tests in this class are expected to fail on the current code.  They
    serve as regression tests once PR #8 is merged.
    """

    def test_battery_emergency_beats_no_other_alerts(self):
        """Baseline: battery emergency alone must return EMERGENCY_LAND."""
        monitor = make_monitor()
        state = make_state(battery_percent=5.0)
        assert monitor.check_state(state) == SafetyAction.EMERGENCY_LAND

    def test_battery_emergency_beats_altitude_slow_down(self):
        """
        Battery emergency (EMERGENCY_LAND) must win over altitude breach (SLOW_DOWN).

        BUG: 'slow_down' > 'emergency_land' lexicographically, so the current
        code incorrectly returns SLOW_DOWN.  This test fails pre-PR #8.
        """
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(battery_percent=5.0, altitude=150.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.EMERGENCY_LAND, (
            f"Expected EMERGENCY_LAND but got {action}. "
            "This failure confirms the SafetyAction string-comparison bug — "
            "see PR #8 for the fix."
        )

    def test_battery_emergency_beats_gps_hover(self):
        """
        Battery emergency (EMERGENCY_LAND) must win over GPS failure (HOVER).

        BUG: 'hover' > 'emergency_land' lexicographically, so the current
        code incorrectly returns HOVER.  This test fails pre-PR #8.
        """
        monitor = make_monitor(min_satellites=6)
        state = make_state(battery_percent=5.0, gps_satellites=3)
        action = monitor.check_state(state)
        assert action == SafetyAction.EMERGENCY_LAND, (
            f"Expected EMERGENCY_LAND but got {action}. "
            "This failure confirms the SafetyAction string-comparison bug — "
            "see PR #8 for the fix."
        )

    def test_battery_critical_beats_altitude_slow_down(self):
        """
        Battery critical (RETURN_HOME) must win over altitude breach (SLOW_DOWN).

        BUG: 'slow_down' > 'return_home' lexicographically, so the current
        code incorrectly returns SLOW_DOWN.  This test fails pre-PR #8.
        """
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(battery_percent=12.0, altitude=150.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME, (
            f"Expected RETURN_HOME but got {action}. "
            "This failure confirms the SafetyAction string-comparison bug — "
            "see PR #8 for the fix."
        )

    def test_return_home_beats_hover(self):
        """
        Signal loss (RETURN_HOME) must beat GPS poor (HOVER).

        BUG: Despite 'return_home' > 'hover' lexicographically (which would give
        the correct answer for Bug 2), Bug 1 still causes a TypeError crash here
        because two alerts fire simultaneously.  Fails pre-PR #8.
        """
        monitor = make_monitor(signal_critical=20.0, min_satellites=6)
        state = make_state(signal_strength=10.0, gps_satellites=3)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME

    def test_all_alerts_battery_emergency_wins(self):
        """
        When every subsystem fires, EMERGENCY_LAND must be the final action.

        BUG: Several string values lexicographically exceed 'emergency_land',
        so the current code will return something else.  Fails pre-PR #8.
        """
        monitor = make_monitor(
            max_altitude=120.0,
            max_distance=500.0,
            max_speed=20.0,
            min_satellites=6,
            signal_critical=20.0,
            max_tilt=45.0,
        )
        state = make_state(
            battery_percent=5.0,   # EMERGENCY_LAND
            altitude=150.0,        # SLOW_DOWN
            distance_from_home=600.0,  # RETURN_HOME (geofence)
            ground_speed=25.0,     # SLOW_DOWN
            gps_satellites=3,      # HOVER
            signal_strength=10.0,  # RETURN_HOME
            pitch=60.0,            # HOVER
        )
        action = monitor.check_state(state)
        assert action == SafetyAction.EMERGENCY_LAND, (
            f"Expected EMERGENCY_LAND but got {action}. "
            "This failure confirms the SafetyAction string-comparison bug — "
            "see PR #8 for the fix."
        )


# ---------------------------------------------------------------------------
# Callback registration and triggering
# ---------------------------------------------------------------------------

class TestCallbacks:

    def test_callback_called_for_each_alert(self):
        """Callbacks fire once per alert.

        BUG: With 2+ alerts, Bug 1 (SafetyLevel not orderable) causes a
        TypeError before callbacks are invoked.  Fails pre-PR #8.
        """
        monitor = make_monitor(max_altitude=120.0, max_speed=20.0)
        state = make_state(altitude=150.0, ground_speed=25.0)

        received = []
        monitor.register_callback(received.append)
        monitor.check_state(state)

        # altitude + speed alerts → 2 callbacks
        assert len(received) == 2

    def test_callback_receives_correct_alert(self):
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=150.0)

        received = []
        monitor.register_callback(received.append)
        monitor.check_state(state)

        assert received[0].source == "altitude"
        assert received[0].level == SafetyLevel.WARNING

    def test_multiple_callbacks_all_called(self):
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=150.0)

        counts = [0, 0]
        monitor.register_callback(lambda _: counts.__setitem__(0, counts[0] + 1))
        monitor.register_callback(lambda _: counts.__setitem__(1, counts[1] + 1))
        monitor.check_state(state)

        assert counts == [1, 1]

    def test_faulty_callback_does_not_crash_monitor(self):
        """A callback that raises an exception must not propagate to the caller."""
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=150.0)

        monitor.register_callback(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
        # Should not raise:
        action = monitor.check_state(state)
        assert action == SafetyAction.SLOW_DOWN

    def test_no_callbacks_when_no_alerts(self):
        monitor = make_monitor()
        state = make_state()

        received = []
        monitor.register_callback(received.append)
        monitor.check_state(state)

        assert received == []


# ---------------------------------------------------------------------------
# get_alerts()
# ---------------------------------------------------------------------------

class TestGetAlerts:

    def test_get_alerts_returns_copy(self):
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=150.0)
        monitor.check_state(state)

        alerts = monitor.get_alerts()
        alerts.clear()
        # Internal list must not be affected
        assert len(monitor.alerts) == 1

    def test_get_alerts_empty_on_nominal(self):
        monitor = make_monitor()
        state = make_state()
        monitor.check_state(state)
        assert monitor.get_alerts() == []


# ---------------------------------------------------------------------------
# SafetyAlert dataclass
# ---------------------------------------------------------------------------

class TestSafetyAlert:

    def test_alert_has_timestamp(self):
        before = time.time()
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=150.0)
        monitor.check_state(state)
        after = time.time()

        alert = monitor.alerts[0]
        assert before <= alert.timestamp <= after

    def test_alert_message_contains_value(self):
        monitor = make_monitor(max_altitude=120.0)
        state = make_state(altitude=999.9)
        monitor.check_state(state)
        assert "999.9" in monitor.alerts[0].message
