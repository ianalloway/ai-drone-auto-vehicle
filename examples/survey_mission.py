#!/usr/bin/env python3
"""
Survey Mission Example
Demonstrates grid-based aerial survey with camera control.
"""

import asyncio
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.communication.mavlink import MAVLinkConnection
from src.sensors.fusion import SensorFusion
from src.decision.safety import SafetyMonitor


@dataclass
class SurveyArea:
    """Define a rectangular survey area."""
    north_lat: float
    south_lat: float
    east_lon: float
    west_lon: float
    altitude: float
    overlap_percent: float = 70.0  # Image overlap percentage
    

@dataclass
class CameraConfig:
    """Camera configuration for survey."""
    sensor_width_mm: float = 6.17  # Typical drone camera sensor
    sensor_height_mm: float = 4.55
    focal_length_mm: float = 4.5
    image_width_px: int = 4000
    image_height_px: int = 3000
    
    def get_ground_coverage(self, altitude_m: float) -> Tuple[float, float]:
        """Calculate ground coverage at given altitude."""
        gsd = (altitude_m * self.sensor_width_mm) / (self.focal_length_mm * self.image_width_px)
        width_m = gsd * self.image_width_px
        height_m = gsd * self.image_height_px
        return width_m, height_m


class SurveyMission:
    """Execute a grid survey mission with photo capture."""
    
    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        self.mavlink = MAVLinkConnection(connection_string)
        self.sensor_fusion = SensorFusion()
        self.safety = SafetyMonitor()
        self.camera = CameraConfig()
        
        self.survey_area: Optional[SurveyArea] = None
        self.waypoints: List[Tuple[float, float, float]] = []
        self.photos_taken = 0
        self.mission_active = False
        
    def set_survey_area(self, north: float, south: float, 
                        east: float, west: float, altitude: float,
                        overlap: float = 70.0):
        """Define the survey area."""
        self.survey_area = SurveyArea(
            north_lat=north,
            south_lat=south,
            east_lon=east,
            west_lon=west,
            altitude=altitude,
            overlap_percent=overlap
        )
        print(f"Survey area set: {north}N to {south}S, {west}W to {east}E @ {altitude}m")
        
    def _generate_grid_waypoints(self) -> List[Tuple[float, float, float]]:
        """Generate lawnmower pattern waypoints for survey."""
        if not self.survey_area:
            return []
            
        area = self.survey_area
        coverage_w, coverage_h = self.camera.get_ground_coverage(area.altitude)
        
        # Calculate spacing based on overlap
        overlap_factor = 1 - (area.overlap_percent / 100)
        spacing_lat = (coverage_h * overlap_factor) / 111320  # meters to degrees
        spacing_lon = (coverage_w * overlap_factor) / 111320
        
        waypoints = []
        lat = area.south_lat
        row = 0
        
        while lat <= area.north_lat:
            if row % 2 == 0:
                # West to East
                lon = area.west_lon
                while lon <= area.east_lon:
                    waypoints.append((lat, lon, area.altitude))
                    lon += spacing_lon
            else:
                # East to West (reverse for efficiency)
                lon = area.east_lon
                while lon >= area.west_lon:
                    waypoints.append((lat, lon, area.altitude))
                    lon -= spacing_lon
                    
            lat += spacing_lat
            row += 1
            
        return waypoints
        
    def plan_survey(self):
        """Plan the survey mission and generate waypoints."""
        self.waypoints = self._generate_grid_waypoints()
        
        # Calculate estimated metrics
        coverage_w, coverage_h = self.camera.get_ground_coverage(self.survey_area.altitude)
        total_photos = len(self.waypoints)
        
        # Estimate flight distance
        total_distance = 0
        for i in range(1, len(self.waypoints)):
            lat1, lon1, _ = self.waypoints[i-1]
            lat2, lon2, _ = self.waypoints[i]
            dlat = (lat2 - lat1) * 111320
            dlon = (lon2 - lon1) * 111320
            total_distance += math.sqrt(dlat**2 + dlon**2)
            
        print(f"\nSurvey Plan:")
        print(f"  Total waypoints: {total_photos}")
        print(f"  Ground coverage per image: {coverage_w:.1f}m x {coverage_h:.1f}m")
        print(f"  Estimated flight distance: {total_distance/1000:.2f} km")
        print(f"  Estimated flight time: {total_distance/5:.0f}s @ 5m/s")
        
        return self.waypoints
        
    async def connect(self):
        """Connect to the vehicle."""
        await self.mavlink.connect()
        print("Connected to vehicle")
        
    async def _capture_photo(self, lat: float, lon: float, alt: float):
        """Capture a photo at current position."""
        # In real implementation, this would trigger camera
        self.photos_taken += 1
        print(f"  Photo {self.photos_taken}: ({lat:.6f}, {lon:.6f}) @ {alt:.1f}m")
        await asyncio.sleep(0.5)  # Simulate capture time
        
    async def execute(self, speed: float = 5.0):
        """Execute the survey mission."""
        if not self.waypoints:
            print("No waypoints - run plan_survey() first")
            return
            
        self.mission_active = True
        self.photos_taken = 0
        
        print(f"\nStarting survey with {len(self.waypoints)} photo points")
        print(f"Flight speed: {speed} m/s")
        
        for i, (lat, lon, alt) in enumerate(self.waypoints):
            if not self.mission_active:
                print("Survey aborted")
                break
                
            # Check safety
            if not self.safety.is_safe():
                print("Safety check failed - holding position")
                await self.mavlink.hold_position()
                await asyncio.sleep(1)
                continue
                
            # Navigate to waypoint
            await self.mavlink.goto(lat, lon, alt, speed=speed)
            
            # Wait to reach waypoint (simplified)
            await asyncio.sleep(0.1)
            
            # Capture photo
            await self._capture_photo(lat, lon, alt)
            
            # Progress update every 10 waypoints
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(self.waypoints)} ({100*(i+1)/len(self.waypoints):.0f}%)")
                
        print(f"\nSurvey complete! {self.photos_taken} photos captured")
        self.mission_active = False
        
    async def abort(self):
        """Abort the survey and hold position."""
        self.mission_active = False
        await self.mavlink.hold_position()
        print("Survey aborted - holding position")


async def main():
    """Example survey mission."""
    survey = SurveyMission()
    
    # Define survey area (small test area)
    survey.set_survey_area(
        north=37.7760,
        south=37.7740,
        east=-122.4170,
        west=-122.4200,
        altitude=50.0,
        overlap=75.0
    )
    
    # Plan the survey
    survey.plan_survey()
    
    try:
        await survey.connect()
        await survey.execute(speed=5.0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        await survey.abort()
    except Exception as e:
        print(f"Error: {e}")
        await survey.abort()


if __name__ == "__main__":
    asyncio.run(main())
