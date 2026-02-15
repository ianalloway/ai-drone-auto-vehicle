"""
MAVLink Communication Module for Drone AI

Provides MAVLink protocol communication for drone control
and telemetry with PX4 and ArduPilot autopilots.
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
from enum import Enum
from loguru import logger


class FlightMode(Enum):
    """Common flight modes."""
    STABILIZE = "STABILIZE"
    ALT_HOLD = "ALT_HOLD"
    LOITER = "LOITER"
    AUTO = "AUTO"
    GUIDED = "GUIDED"
    RTL = "RTL"
    LAND = "LAND"
    OFFBOARD = "OFFBOARD"


@dataclass
class Telemetry:
    """Drone telemetry data."""
    latitude: float
    longitude: float
    altitude: float
    relative_altitude: float
    heading: float
    ground_speed: float
    vertical_speed: float
    battery_voltage: float
    battery_percent: float
    armed: bool
    flight_mode: str
    gps_satellites: int
    roll: float
    pitch: float
    yaw: float
    timestamp: float


class MAVLinkController:
    """
    MAVLink communication controller for drone operations.
    
    Supports connection to PX4 and ArduPilot autopilots via
    serial, UDP, or TCP connections.
    """
    
    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14550",
        baudrate: int = 57600,
        source_system: int = 255,
        source_component: int = 0
    ):
        """
        Initialize MAVLink controller.
        
        Args:
            connection_string: Connection string (e.g., "udp:127.0.0.1:14550")
            baudrate: Baud rate for serial connections
            source_system: MAVLink source system ID
            source_component: MAVLink source component ID
        """
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.source_system = source_system
        self.source_component = source_component
        
        self.connection = None
        self.vehicle = None
        self._connected = False
        self._armed = False
        
        self.telemetry_callbacks: List[Callable[[Telemetry], None]] = []
        
        # Home position
        self.home_position: Optional[Tuple[float, float, float]] = None
        
        logger.info(f"MAVLinkController initialized: {connection_string}")
    
    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to the vehicle.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        try:
            from pymavlink import mavutil
            
            logger.info(f"Connecting to {self.connection_string}...")
            
            self.connection = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baudrate,
                source_system=self.source_system,
                source_component=self.source_component
            )
            
            # Wait for heartbeat
            msg = self.connection.wait_heartbeat(timeout=timeout)
            if msg:
                self._connected = True
                logger.info(f"Connected to vehicle (system {self.connection.target_system})")
                return True
            else:
                logger.error("Connection timeout - no heartbeat received")
                return False
                
        except ImportError:
            logger.warning("pymavlink not installed, using mock connection")
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the vehicle."""
        if self.connection:
            self.connection.close()
        self._connected = False
        logger.info("Disconnected from vehicle")
    
    def arm(self, force: bool = False) -> bool:
        """
        Arm the vehicle.
        
        Args:
            force: Force arming even if pre-arm checks fail
            
        Returns:
            True if armed successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            if self.connection:
                self.connection.arducopter_arm()
                # Wait for arm confirmation
                time.sleep(1)
            
            self._armed = True
            logger.info("Vehicle armed")
            return True
            
        except Exception as e:
            logger.error(f"Arm failed: {e}")
            return False
    
    def disarm(self) -> bool:
        """
        Disarm the vehicle.
        
        Returns:
            True if disarmed successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            if self.connection:
                self.connection.arducopter_disarm()
            
            self._armed = False
            logger.info("Vehicle disarmed")
            return True
            
        except Exception as e:
            logger.error(f"Disarm failed: {e}")
            return False
    
    def takeoff(self, altitude: float) -> bool:
        """
        Command vehicle to takeoff.
        
        Args:
            altitude: Target altitude in meters
            
        Returns:
            True if takeoff command sent successfully
        """
        if not self._armed:
            logger.error("Vehicle not armed")
            return False
        
        try:
            if self.connection:
                self.connection.mav.command_long_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    22,  # MAV_CMD_NAV_TAKEOFF
                    0,   # confirmation
                    0, 0, 0, 0,  # params 1-4
                    0, 0,  # lat, lon (use current)
                    altitude  # altitude
                )
            
            logger.info(f"Takeoff commanded to {altitude}m")
            return True
            
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False
    
    def land(self) -> bool:
        """
        Command vehicle to land.
        
        Returns:
            True if land command sent successfully
        """
        try:
            if self.connection:
                self.connection.mav.command_long_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    21,  # MAV_CMD_NAV_LAND
                    0,   # confirmation
                    0, 0, 0, 0,  # params 1-4
                    0, 0, 0  # lat, lon, alt (use current)
                )
            
            logger.info("Land commanded")
            return True
            
        except Exception as e:
            logger.error(f"Land failed: {e}")
            return False
    
    def goto(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        ground_speed: float = 5.0
    ) -> bool:
        """
        Command vehicle to go to a position.
        
        Args:
            latitude: Target latitude
            longitude: Target longitude
            altitude: Target altitude (meters, relative to home)
            ground_speed: Ground speed (m/s)
            
        Returns:
            True if command sent successfully
        """
        try:
            if self.connection:
                self.connection.mav.mission_item_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    0,  # seq
                    0,  # frame (global relative alt)
                    16, # MAV_CMD_NAV_WAYPOINT
                    2,  # current (guided mode)
                    1,  # autocontinue
                    0, 0, 0, 0,  # params
                    int(latitude * 1e7),
                    int(longitude * 1e7),
                    altitude
                )
            
            logger.info(f"Goto commanded: {latitude}, {longitude}, {altitude}m")
            return True
            
        except Exception as e:
            logger.error(f"Goto failed: {e}")
            return False
    
    def set_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float = 0.0
    ) -> bool:
        """
        Set velocity in body frame.
        
        Args:
            vx: Forward velocity (m/s)
            vy: Right velocity (m/s)
            vz: Down velocity (m/s)
            yaw_rate: Yaw rate (rad/s)
            
        Returns:
            True if command sent successfully
        """
        try:
            if self.connection:
                self.connection.mav.set_position_target_local_ned_send(
                    0,  # time_boot_ms
                    self.connection.target_system,
                    self.connection.target_component,
                    8,  # MAV_FRAME_BODY_NED
                    0b0000111111000111,  # type_mask (velocity only)
                    0, 0, 0,  # position (ignored)
                    vx, vy, vz,  # velocity
                    0, 0, 0,  # acceleration (ignored)
                    0, yaw_rate  # yaw, yaw_rate
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Set velocity failed: {e}")
            return False
    
    def set_mode(self, mode: FlightMode) -> bool:
        """
        Set flight mode.
        
        Args:
            mode: Target flight mode
            
        Returns:
            True if mode change successful
        """
        try:
            if self.connection:
                mode_mapping = self.connection.mode_mapping()
                if mode.value in mode_mapping:
                    mode_id = mode_mapping[mode.value]
                    self.connection.set_mode(mode_id)
                    logger.info(f"Mode set to {mode.value}")
                    return True
            
            logger.info(f"Mode set to {mode.value} (mock)")
            return True
            
        except Exception as e:
            logger.error(f"Set mode failed: {e}")
            return False
    
    def return_to_launch(self) -> bool:
        """
        Command vehicle to return to launch position.
        
        Returns:
            True if RTL command sent successfully
        """
        return self.set_mode(FlightMode.RTL)
    
    def get_telemetry(self) -> Optional[Telemetry]:
        """
        Get current telemetry data.
        
        Returns:
            Telemetry data or None if not available
        """
        if not self._connected:
            return None
        
        try:
            if self.connection:
                # Request data streams
                msg = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
                att = self.connection.recv_match(type='ATTITUDE', blocking=False)
                bat = self.connection.recv_match(type='SYS_STATUS', blocking=False)
                
                # Build telemetry from available messages
                return Telemetry(
                    latitude=msg.lat / 1e7 if msg else 0,
                    longitude=msg.lon / 1e7 if msg else 0,
                    altitude=msg.alt / 1000 if msg else 0,
                    relative_altitude=msg.relative_alt / 1000 if msg else 0,
                    heading=msg.hdg / 100 if msg else 0,
                    ground_speed=0,
                    vertical_speed=msg.vz / 100 if msg else 0,
                    battery_voltage=bat.voltage_battery / 1000 if bat else 0,
                    battery_percent=bat.battery_remaining if bat else 100,
                    armed=self._armed,
                    flight_mode="UNKNOWN",
                    gps_satellites=0,
                    roll=att.roll if att else 0,
                    pitch=att.pitch if att else 0,
                    yaw=att.yaw if att else 0,
                    timestamp=time.time()
                )
            
            # Mock telemetry
            return Telemetry(
                latitude=37.7749,
                longitude=-122.4194,
                altitude=50.0,
                relative_altitude=50.0,
                heading=0.0,
                ground_speed=0.0,
                vertical_speed=0.0,
                battery_voltage=12.6,
                battery_percent=85.0,
                armed=self._armed,
                flight_mode="GUIDED",
                gps_satellites=12,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Get telemetry failed: {e}")
            return None
    
    def register_telemetry_callback(self, callback: Callable[[Telemetry], None]):
        """Register callback for telemetry updates."""
        self.telemetry_callbacks.append(callback)
    
    def is_connected(self) -> bool:
        """Check if connected to vehicle."""
        return self._connected
    
    def is_armed(self) -> bool:
        """Check if vehicle is armed."""
        return self._armed
