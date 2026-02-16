#!/usr/bin/env python3
"""
Waypoint Mission Example
========================

This example demonstrates how to create a simple waypoint mission
where the drone flies to a series of GPS coordinates.

Usage:
    python waypoint_mission.py

Requirements:
    - DroneAI framework installed
    - Connected to drone via MAVLink
    - GPS lock acquired
"""

import asyncio
from typing import List, Tuple

# Simulated DroneAI imports (replace with actual imports when framework is complete)
# from drone_ai import Drone, Mission, Waypoint
# from drone_ai.navigation import GPSCoordinate


class GPSCoordinate:
    """GPS coordinate with latitude, longitude, and altitude."""
    
    def __init__(self, lat: float, lon: float, alt: float = 10.0):
        self.lat = lat
        self.lon = lon
        self.alt = alt
    
    def __repr__(self):
        return f"GPS({self.lat:.6f}, {self.lon:.6f}, {self.alt}m)"


class Waypoint:
    """A waypoint in a mission."""
    
    def __init__(self, coord: GPSCoordinate, speed: float = 5.0, hold_time: float = 0.0):
        self.coord = coord
        self.speed = speed  # m/s
        self.hold_time = hold_time  # seconds to hover at waypoint
    
    def __repr__(self):
        return f"Waypoint({self.coord}, speed={self.speed}m/s, hold={self.hold_time}s)"


class WaypointMission:
    """
    A simple waypoint mission that flies the drone through a series of GPS coordinates.
    
    Features:
    - Configurable speed between waypoints
    - Hold time at each waypoint
    - Return-to-launch (RTL) option
    - Altitude management
    """
    
    def __init__(self, name: str = "Waypoint Mission"):
        self.name = name
        self.waypoints: List[Waypoint] = []
        self.return_to_launch = True
        self.default_altitude = 10.0  # meters
        self.default_speed = 5.0  # m/s
    
    def add_waypoint(self, lat: float, lon: float, alt: float = None, 
                     speed: float = None, hold_time: float = 0.0):
        """Add a waypoint to the mission."""
        coord = GPSCoordinate(
            lat, lon, 
            alt if alt is not None else self.default_altitude
        )
        waypoint = Waypoint(
            coord,
            speed if speed is not None else self.default_speed,
            hold_time
        )
        self.waypoints.append(waypoint)
        print(f"Added {waypoint}")
        return self
    
    def add_waypoints(self, coords: List[Tuple[float, float, float]]):
        """Add multiple waypoints at once."""
        for coord in coords:
            if len(coord) == 2:
                self.add_waypoint(coord[0], coord[1])
            else:
                self.add_waypoint(coord[0], coord[1], coord[2])
        return self
    
    def set_return_to_launch(self, rtl: bool):
        """Enable or disable return-to-launch after mission completion."""
        self.return_to_launch = rtl
        return self
    
    def calculate_distance(self) -> float:
        """Calculate total mission distance in meters (approximate)."""
        if len(self.waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            # Simplified distance calculation (Haversine would be more accurate)
            w1, w2 = self.waypoints[i], self.waypoints[i + 1]
            lat_diff = (w2.coord.lat - w1.coord.lat) * 111000  # ~111km per degree
            lon_diff = (w2.coord.lon - w1.coord.lon) * 111000 * 0.85  # Adjust for latitude
            alt_diff = w2.coord.alt - w1.coord.alt
            total += (lat_diff**2 + lon_diff**2 + alt_diff**2) ** 0.5
        
        return total
    
    def estimate_duration(self) -> float:
        """Estimate mission duration in seconds."""
        distance = self.calculate_distance()
        avg_speed = sum(w.speed for w in self.waypoints) / len(self.waypoints) if self.waypoints else self.default_speed
        hold_time = sum(w.hold_time for w in self.waypoints)
        
        return (distance / avg_speed) + hold_time
    
    async def execute(self):
        """Execute the mission (simulation)."""
        print(f"\n{'='*50}")
        print(f"Executing Mission: {self.name}")
        print(f"Waypoints: {len(self.waypoints)}")
        print(f"Estimated Distance: {self.calculate_distance():.1f}m")
        print(f"Estimated Duration: {self.estimate_duration():.1f}s")
        print(f"Return to Launch: {self.return_to_launch}")
        print(f"{'='*50}\n")
        
        # Simulate takeoff
        print("Taking off...")
        await asyncio.sleep(1)
        print(f"Reached altitude: {self.default_altitude}m\n")
        
        # Navigate to each waypoint
        for i, waypoint in enumerate(self.waypoints, 1):
            print(f"Navigating to waypoint {i}/{len(self.waypoints)}: {waypoint.coord}")
            await asyncio.sleep(0.5)  # Simulate flight time
            print(f"  Arrived at waypoint {i}")
            
            if waypoint.hold_time > 0:
                print(f"  Holding for {waypoint.hold_time}s...")
                await asyncio.sleep(0.2)  # Simulate hold
        
        # Return to launch
        if self.return_to_launch:
            print("\nReturning to launch point...")
            await asyncio.sleep(0.5)
            print("Arrived at launch point")
        
        # Land
        print("Landing...")
        await asyncio.sleep(0.5)
        print("Mission complete!\n")


async def main():
    """Example: Create and execute a simple waypoint mission."""
    
    # Create a mission
    mission = WaypointMission("Survey Flight")
    
    # Define waypoints (example coordinates - San Francisco area)
    # In a real mission, these would be actual GPS coordinates
    mission.add_waypoint(37.7749, -122.4194, alt=15, hold_time=2)  # Start point
    mission.add_waypoint(37.7750, -122.4180, alt=15)  # East
    mission.add_waypoint(37.7760, -122.4180, alt=20)  # North-East
    mission.add_waypoint(37.7760, -122.4194, alt=20, hold_time=5)  # North (photo point)
    mission.add_waypoint(37.7749, -122.4194, alt=15)  # Return to start
    
    # Configure mission
    mission.set_return_to_launch(True)
    
    # Execute
    await mission.execute()


if __name__ == "__main__":
    print("Drone AI - Waypoint Mission Example")
    print("====================================\n")
    asyncio.run(main())
