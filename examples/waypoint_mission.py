#!/usr/bin/env python3
"""
Waypoint Mission Example
Demonstrates autonomous waypoint navigation with obstacle avoidance.
"""

import asyncio
from dataclasses import dataclass
from typing import List

from src.planning.astar import AStarPlanner
from src.planning.avoidance import ObstacleAvoider
from src.sensors.fusion import SensorFusion, GPSData, IMUData
from src.communication.mavlink import MAVLinkConnection
from src.decision.safety import SafetyMonitor


@dataclass
class Waypoint:
    """A 3D waypoint with optional loiter time."""
    latitude: float
    longitude: float
    altitude: float
    loiter_time: float = 0.0  # seconds to hover at waypoint
    acceptance_radius: float = 2.0  # meters


class WaypointMission:
    """Execute a series of waypoints with obstacle avoidance."""
    
    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        self.mavlink = MAVLinkConnection(connection_string)
        self.planner = AStarPlanner(grid_size=100, safety_margin=3.0)
        self.avoider = ObstacleAvoider(safety_distance=5.0)
        self.sensor_fusion = SensorFusion()
        self.safety = SafetyMonitor()
        
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        self.mission_active = False
        
    def add_waypoint(self, lat: float, lon: float, alt: float, 
                     loiter: float = 0.0, radius: float = 2.0):
        """Add a waypoint to the mission."""
        wp = Waypoint(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            loiter_time=loiter,
            acceptance_radius=radius
        )
        self.waypoints.append(wp)
        print(f"Added waypoint {len(self.waypoints)}: ({lat}, {lon}) @ {alt}m")
        
    def clear_waypoints(self):
        """Clear all waypoints."""
        self.waypoints.clear()
        self.current_waypoint_idx = 0
        print("Cleared all waypoints")
        
    async def connect(self):
        """Connect to the vehicle."""
        await self.mavlink.connect()
        print("Connected to vehicle")
        
    async def arm_and_takeoff(self, target_altitude: float):
        """Arm the vehicle and takeoff to target altitude."""
        print(f"Arming and taking off to {target_altitude}m...")
        
        # Wait for vehicle to be armable
        while not self.mavlink.is_armable():
            print("Waiting for vehicle to be armable...")
            await asyncio.sleep(1)
            
        # Arm the vehicle
        await self.mavlink.arm()
        print("Vehicle armed")
        
        # Takeoff
        await self.mavlink.takeoff(target_altitude)
        
        # Wait until target altitude reached
        while True:
            state = self.sensor_fusion.get_state()
            if state and state.altitude >= target_altitude * 0.95:
                print(f"Reached target altitude: {state.altitude:.1f}m")
                break
            await asyncio.sleep(0.5)
            
    def _distance_to_waypoint(self, wp: Waypoint) -> float:
        """Calculate distance to waypoint in meters."""
        state = self.sensor_fusion.get_state()
        if not state:
            return float('inf')
            
        # Simplified distance calculation (should use haversine for real impl)
        dlat = (wp.latitude - state.latitude) * 111320
        dlon = (wp.longitude - state.longitude) * 111320
        dalt = wp.altitude - state.altitude
        
        return (dlat**2 + dlon**2 + dalt**2) ** 0.5
        
    async def _navigate_to_waypoint(self, wp: Waypoint):
        """Navigate to a single waypoint with obstacle avoidance."""
        print(f"Navigating to ({wp.latitude}, {wp.longitude}) @ {wp.altitude}m")
        
        while self._distance_to_waypoint(wp) > wp.acceptance_radius:
            # Check safety
            if not self.safety.is_safe():
                print("Safety check failed - holding position")
                await self.mavlink.hold_position()
                await asyncio.sleep(1)
                continue
                
            # Check for obstacles and adjust path if needed
            obstacles = self.avoider.get_nearby_obstacles()
            if obstacles:
                # Calculate avoidance velocity
                avoidance_vel = self.avoider.calculate_avoidance_velocity(
                    current_velocity=(0, 0, 0),
                    obstacles=obstacles
                )
                if avoidance_vel:
                    print(f"Avoiding obstacle, adjusting velocity")
                    await self.mavlink.set_velocity(*avoidance_vel)
                    await asyncio.sleep(0.1)
                    continue
                    
            # Normal navigation to waypoint
            await self.mavlink.goto(wp.latitude, wp.longitude, wp.altitude)
            await asyncio.sleep(0.1)
            
        print(f"Reached waypoint")
        
        # Loiter if specified
        if wp.loiter_time > 0:
            print(f"Loitering for {wp.loiter_time}s")
            await asyncio.sleep(wp.loiter_time)
            
    async def execute(self):
        """Execute the waypoint mission."""
        if not self.waypoints:
            print("No waypoints defined")
            return
            
        self.mission_active = True
        self.current_waypoint_idx = 0
        
        print(f"Starting mission with {len(self.waypoints)} waypoints")
        
        for i, wp in enumerate(self.waypoints):
            if not self.mission_active:
                print("Mission aborted")
                break
                
            self.current_waypoint_idx = i
            print(f"\n--- Waypoint {i+1}/{len(self.waypoints)} ---")
            await self._navigate_to_waypoint(wp)
            
        print("\nMission complete!")
        self.mission_active = False
        
    async def abort(self):
        """Abort the current mission and hold position."""
        self.mission_active = False
        await self.mavlink.hold_position()
        print("Mission aborted - holding position")
        
    async def return_to_launch(self):
        """Return to launch position and land."""
        print("Returning to launch...")
        await self.mavlink.return_to_launch()


async def main():
    """Example waypoint mission."""
    mission = WaypointMission()
    
    # Define waypoints (example coordinates)
    mission.add_waypoint(37.7749, -122.4194, 30.0, loiter=5.0)  # San Francisco
    mission.add_waypoint(37.7751, -122.4180, 35.0)
    mission.add_waypoint(37.7755, -122.4170, 30.0, loiter=3.0)
    mission.add_waypoint(37.7749, -122.4194, 30.0)  # Return to start
    
    try:
        await mission.connect()
        await mission.arm_and_takeoff(30.0)
        await mission.execute()
        await mission.return_to_launch()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        await mission.abort()
    except Exception as e:
        print(f"Error: {e}")
        await mission.abort()


if __name__ == "__main__":
    asyncio.run(main())
