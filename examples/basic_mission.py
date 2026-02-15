#!/usr/bin/env python3
"""
Basic Mission Example for Drone AI

Demonstrates a simple autonomous mission with waypoint navigation
and obstacle avoidance.
"""

import time
import numpy as np
from loguru import logger

# Import Drone AI modules
import sys
sys.path.insert(0, '..')

from src.communication.mavlink import MAVLinkController, FlightMode
from src.vision.detection import ObjectDetector
from src.planning.astar import AStarPlanner
from src.decision.safety import SafetyMonitor, DroneState


def main():
    """Run a basic autonomous mission."""
    
    # Configuration
    CONNECTION = "udp:127.0.0.1:14550"
    TAKEOFF_ALTITUDE = 10.0  # meters
    WAYPOINTS = [
        (37.7749, -122.4194, 10),  # Start
        (37.7759, -122.4184, 15),  # Waypoint 1
        (37.7769, -122.4174, 20),  # Waypoint 2
        (37.7749, -122.4194, 10),  # Return to start
    ]
    
    logger.info("Starting basic mission example")
    
    # Initialize components
    controller = MAVLinkController(CONNECTION)
    detector = ObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.5)
    safety = SafetyMonitor(
        battery_warning=30,
        battery_critical=15,
        max_altitude=50,
        max_distance=200
    )
    
    # Connect to vehicle
    logger.info("Connecting to vehicle...")
    if not controller.connect(timeout=30):
        logger.error("Failed to connect to vehicle")
        return
    
    logger.info("Connected! Initializing vision system...")
    detector.initialize()
    
    # Set home position for safety monitor
    safety.set_geofence((37.7749, -122.4194), 200)
    
    try:
        # Arm and takeoff
        logger.info("Arming vehicle...")
        if not controller.arm():
            logger.error("Failed to arm")
            return
        
        logger.info(f"Taking off to {TAKEOFF_ALTITUDE}m...")
        controller.set_mode(FlightMode.GUIDED)
        controller.takeoff(TAKEOFF_ALTITUDE)
        
        # Wait for takeoff
        time.sleep(5)
        
        # Execute waypoint mission
        for i, (lat, lon, alt) in enumerate(WAYPOINTS):
            logger.info(f"Navigating to waypoint {i+1}: ({lat}, {lon}, {alt}m)")
            
            controller.goto(lat, lon, alt)
            
            # Monitor progress
            while True:
                # Get telemetry
                telemetry = controller.get_telemetry()
                if telemetry is None:
                    logger.warning("No telemetry")
                    time.sleep(1)
                    continue
                
                # Check safety
                state = DroneState(
                    battery_percent=telemetry.battery_percent,
                    altitude=telemetry.relative_altitude,
                    ground_speed=telemetry.ground_speed,
                    vertical_speed=telemetry.vertical_speed,
                    distance_from_home=0,  # Calculate from position
                    signal_strength=100,
                    gps_satellites=telemetry.gps_satellites,
                    is_armed=telemetry.armed,
                    flight_mode=telemetry.flight_mode,
                    heading=telemetry.heading,
                    pitch=telemetry.pitch,
                    roll=telemetry.roll
                )
                
                action = safety.check_state(state)
                
                if action.value >= 3:  # RETURN_HOME or worse
                    logger.warning(f"Safety action required: {action}")
                    controller.return_to_launch()
                    break
                
                # Check if waypoint reached (simplified)
                # In real implementation, calculate distance to waypoint
                time.sleep(2)
                break  # Move to next waypoint for demo
        
        # Land
        logger.info("Mission complete, landing...")
        controller.land()
        
        # Wait for landing
        time.sleep(10)
        
        # Disarm
        controller.disarm()
        logger.info("Mission completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Mission interrupted by user")
        controller.land()
        time.sleep(5)
        controller.disarm()
    
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        controller.return_to_launch()
    
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
