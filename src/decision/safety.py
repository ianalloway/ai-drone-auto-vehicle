"""
Safety Monitor Module for Drone AI

Provides safety monitoring and emergency response systems
for autonomous drone operations.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Callable
from loguru import logger
import time


class SafetyLevel(Enum):
    """Safety alert levels."""
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


class SafetyAction(Enum):
    """Actions to take in response to safety events."""
    CONTINUE = "continue"
    SLOW_DOWN = "slow_down"
    HOVER = "hover"
    RETURN_HOME = "return_home"
    EMERGENCY_LAND = "emergency_land"
    KILL_MOTORS = "kill_motors"


@dataclass
class SafetyAlert:
    """Represents a safety alert."""
    level: SafetyLevel
    source: str
    message: str
    action: SafetyAction
    timestamp: float


@dataclass
class DroneState:
    """Current state of the drone for safety monitoring."""
    battery_percent: float
    altitude: float
    ground_speed: float
    vertical_speed: float
    distance_from_home: float
    signal_strength: float
    gps_satellites: int
    is_armed: bool
    flight_mode: str
    heading: float
    pitch: float
    roll: float


class SafetyMonitor:
    """
    Monitors drone state and enforces safety constraints.
    
    Provides real-time safety monitoring with configurable thresholds
    and automatic emergency response.
    """
    
    def __init__(
        self,
        battery_warning: float = 30.0,
        battery_critical: float = 15.0,
        battery_emergency: float = 10.0,
        max_altitude: float = 120.0,
        max_distance: float = 500.0,
        max_speed: float = 20.0,
        min_satellites: int = 6,
        signal_warning: float = 40.0,
        signal_critical: float = 20.0,
        max_tilt: float = 45.0
    ):
        """
        Initialize safety monitor with thresholds.
        
        Args:
            battery_warning: Battery level for warning
            battery_critical: Battery level for critical alert
            battery_emergency: Battery level for emergency
            max_altitude: Maximum allowed altitude (meters)
            max_distance: Maximum distance from home (meters)
            max_speed: Maximum allowed speed (m/s)
            min_satellites: Minimum GPS satellites required
            signal_warning: Signal strength for warning
            signal_critical: Signal strength for critical alert
            max_tilt: Maximum allowed tilt angle (degrees)
        """
        self.battery_warning = battery_warning
        self.battery_critical = battery_critical
        self.battery_emergency = battery_emergency
        self.max_altitude = max_altitude
        self.max_distance = max_distance
        self.max_speed = max_speed
        self.min_satellites = min_satellites
        self.signal_warning = signal_warning
        self.signal_critical = signal_critical
        self.max_tilt = max_tilt
        
        self.alerts: List[SafetyAlert] = []
        self.current_level = SafetyLevel.NORMAL
        self.callbacks: List[Callable[[SafetyAlert], None]] = []
        
        self.geofence_enabled = True
        self.geofence_center = (0.0, 0.0)
        self.geofence_radius = max_distance
        
        logger.info("SafetyMonitor initialized")
    
    def register_callback(self, callback: Callable[[SafetyAlert], None]):
        """Register callback for safety alerts."""
        self.callbacks.append(callback)
    
    def check_state(self, state: DroneState) -> SafetyAction:
        """
        Check drone state and return required action.
        
        Args:
            state: Current drone state
            
        Returns:
            SafetyAction to take
        """
        self.alerts.clear()
        action = SafetyAction.CONTINUE
        
        # Battery checks
        battery_action = self._check_battery(state.battery_percent)
        if battery_action.value > action.value:
            action = battery_action
        
        # Altitude check
        altitude_action = self._check_altitude(state.altitude)
        if altitude_action.value > action.value:
            action = altitude_action
        
        # Distance check
        distance_action = self._check_distance(state.distance_from_home)
        if distance_action.value > action.value:
            action = distance_action
        
        # Speed check
        speed_action = self._check_speed(state.ground_speed)
        if speed_action.value > action.value:
            action = speed_action
        
        # GPS check
        gps_action = self._check_gps(state.gps_satellites)
        if gps_action.value > action.value:
            action = gps_action
        
        # Signal check
        signal_action = self._check_signal(state.signal_strength)
        if signal_action.value > action.value:
            action = signal_action
        
        # Attitude check
        attitude_action = self._check_attitude(state.pitch, state.roll)
        if attitude_action.value > action.value:
            action = attitude_action
        
        # Update current safety level
        if self.alerts:
            self.current_level = max(a.level for a in self.alerts)
        else:
            self.current_level = SafetyLevel.NORMAL
        
        # Trigger callbacks
        for alert in self.alerts:
            for callback in self.callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        return action
    
    def _check_battery(self, battery: float) -> SafetyAction:
        """Check battery level."""
        if battery <= self.battery_emergency:
            self._add_alert(
                SafetyLevel.EMERGENCY,
                "battery",
                f"Battery critical: {battery:.1f}%",
                SafetyAction.EMERGENCY_LAND
            )
            return SafetyAction.EMERGENCY_LAND
        
        if battery <= self.battery_critical:
            self._add_alert(
                SafetyLevel.CRITICAL,
                "battery",
                f"Battery low: {battery:.1f}%",
                SafetyAction.RETURN_HOME
            )
            return SafetyAction.RETURN_HOME
        
        if battery <= self.battery_warning:
            self._add_alert(
                SafetyLevel.WARNING,
                "battery",
                f"Battery warning: {battery:.1f}%",
                SafetyAction.CONTINUE
            )
        
        return SafetyAction.CONTINUE
    
    def _check_altitude(self, altitude: float) -> SafetyAction:
        """Check altitude limits."""
        if altitude > self.max_altitude:
            self._add_alert(
                SafetyLevel.WARNING,
                "altitude",
                f"Altitude exceeded: {altitude:.1f}m (max: {self.max_altitude}m)",
                SafetyAction.SLOW_DOWN
            )
            return SafetyAction.SLOW_DOWN
        
        return SafetyAction.CONTINUE
    
    def _check_distance(self, distance: float) -> SafetyAction:
        """Check distance from home."""
        if self.geofence_enabled and distance > self.geofence_radius:
            self._add_alert(
                SafetyLevel.CRITICAL,
                "geofence",
                f"Geofence breach: {distance:.1f}m from home",
                SafetyAction.RETURN_HOME
            )
            return SafetyAction.RETURN_HOME
        
        if distance > self.max_distance * 0.9:
            self._add_alert(
                SafetyLevel.CAUTION,
                "distance",
                f"Approaching max distance: {distance:.1f}m",
                SafetyAction.SLOW_DOWN
            )
            return SafetyAction.SLOW_DOWN
        
        return SafetyAction.CONTINUE
    
    def _check_speed(self, speed: float) -> SafetyAction:
        """Check speed limits."""
        if speed > self.max_speed:
            self._add_alert(
                SafetyLevel.WARNING,
                "speed",
                f"Speed exceeded: {speed:.1f}m/s (max: {self.max_speed}m/s)",
                SafetyAction.SLOW_DOWN
            )
            return SafetyAction.SLOW_DOWN
        
        return SafetyAction.CONTINUE
    
    def _check_gps(self, satellites: int) -> SafetyAction:
        """Check GPS quality."""
        if satellites < self.min_satellites:
            self._add_alert(
                SafetyLevel.WARNING,
                "gps",
                f"Low GPS quality: {satellites} satellites",
                SafetyAction.HOVER
            )
            return SafetyAction.HOVER
        
        return SafetyAction.CONTINUE
    
    def _check_signal(self, signal: float) -> SafetyAction:
        """Check signal strength."""
        if signal < self.signal_critical:
            self._add_alert(
                SafetyLevel.CRITICAL,
                "signal",
                f"Signal lost: {signal:.1f}%",
                SafetyAction.RETURN_HOME
            )
            return SafetyAction.RETURN_HOME
        
        if signal < self.signal_warning:
            self._add_alert(
                SafetyLevel.WARNING,
                "signal",
                f"Signal weak: {signal:.1f}%",
                SafetyAction.SLOW_DOWN
            )
            return SafetyAction.SLOW_DOWN
        
        return SafetyAction.CONTINUE
    
    def _check_attitude(self, pitch: float, roll: float) -> SafetyAction:
        """Check attitude (tilt angles)."""
        max_tilt = max(abs(pitch), abs(roll))
        
        if max_tilt > self.max_tilt:
            self._add_alert(
                SafetyLevel.CRITICAL,
                "attitude",
                f"Excessive tilt: {max_tilt:.1f} degrees",
                SafetyAction.HOVER
            )
            return SafetyAction.HOVER
        
        return SafetyAction.CONTINUE
    
    def _add_alert(
        self,
        level: SafetyLevel,
        source: str,
        message: str,
        action: SafetyAction
    ):
        """Add a safety alert."""
        alert = SafetyAlert(
            level=level,
            source=source,
            message=message,
            action=action,
            timestamp=time.time()
        )
        self.alerts.append(alert)
        
        log_func = {
            SafetyLevel.NORMAL: logger.debug,
            SafetyLevel.CAUTION: logger.info,
            SafetyLevel.WARNING: logger.warning,
            SafetyLevel.CRITICAL: logger.error,
            SafetyLevel.EMERGENCY: logger.critical
        }.get(level, logger.info)
        
        log_func(f"Safety alert [{level.name}]: {message}")
    
    def set_geofence(self, center: tuple, radius: float):
        """Set geofence parameters."""
        self.geofence_center = center
        self.geofence_radius = radius
        logger.info(f"Geofence set: center={center}, radius={radius}m")
    
    def disable_geofence(self):
        """Disable geofence."""
        self.geofence_enabled = False
        logger.warning("Geofence disabled")
    
    def get_alerts(self) -> List[SafetyAlert]:
        """Get current alerts."""
        return self.alerts.copy()
