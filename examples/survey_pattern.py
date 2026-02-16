#!/usr/bin/env python3
"""
Survey Pattern Mission Example
==============================

This example demonstrates how to create automated survey patterns
for mapping, inspection, or search operations.

Patterns included:
- Grid/Lawnmower pattern
- Spiral pattern
- Expanding square search

Usage:
    python survey_pattern.py

Requirements:
    - DroneAI framework installed
    - Connected to drone via MAVLink
    - GPS lock acquired
"""

import asyncio
import math
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    GRID = "grid"
    SPIRAL = "spiral"
    EXPANDING_SQUARE = "expanding_square"


@dataclass
class SurveyConfig:
    """Configuration for survey missions."""
    center_lat: float
    center_lon: float
    altitude: float = 20.0  # meters
    area_width: float = 100.0  # meters
    area_height: float = 100.0  # meters
    line_spacing: float = 10.0  # meters (for overlap in photos)
    speed: float = 3.0  # m/s
    camera_trigger: bool = True
    camera_interval: float = 2.0  # seconds between photos


class SurveyPatternGenerator:
    """
    Generates flight paths for various survey patterns.
    
    All coordinates are returned as (lat, lon, alt) tuples.
    """
    
    # Approximate meters per degree at mid-latitudes
    METERS_PER_DEG_LAT = 111000
    METERS_PER_DEG_LON = 85000  # Varies with latitude
    
    @classmethod
    def meters_to_degrees(cls, meters_lat: float, meters_lon: float) -> Tuple[float, float]:
        """Convert meters to approximate degrees."""
        return (
            meters_lat / cls.METERS_PER_DEG_LAT,
            meters_lon / cls.METERS_PER_DEG_LON
        )
    
    @classmethod
    def generate_grid_pattern(cls, config: SurveyConfig) -> List[Tuple[float, float, float]]:
        """
        Generate a grid/lawnmower survey pattern.
        
        This pattern is ideal for:
        - Aerial mapping/photogrammetry
        - Agricultural surveys
        - Search and rescue grid searches
        
        Returns list of (lat, lon, alt) waypoints.
        """
        waypoints = []
        
        # Calculate number of lines needed
        num_lines = int(config.area_height / config.line_spacing) + 1
        
        # Convert dimensions to degrees
        half_width_deg = (config.area_width / 2) / cls.METERS_PER_DEG_LON
        half_height_deg = (config.area_height / 2) / cls.METERS_PER_DEG_LAT
        line_spacing_deg = config.line_spacing / cls.METERS_PER_DEG_LAT
        
        # Starting corner (bottom-left of survey area)
        start_lat = config.center_lat - half_height_deg
        start_lon = config.center_lon - half_width_deg
        
        # Generate lawnmower pattern
        for i in range(num_lines):
            lat = start_lat + (i * line_spacing_deg)
            
            if i % 2 == 0:
                # Left to right
                waypoints.append((lat, start_lon, config.altitude))
                waypoints.append((lat, start_lon + (2 * half_width_deg), config.altitude))
            else:
                # Right to left
                waypoints.append((lat, start_lon + (2 * half_width_deg), config.altitude))
                waypoints.append((lat, start_lon, config.altitude))
        
        return waypoints
    
    @classmethod
    def generate_spiral_pattern(cls, config: SurveyConfig, inward: bool = True) -> List[Tuple[float, float, float]]:
        """
        Generate a spiral survey pattern.
        
        This pattern is ideal for:
        - Point-of-interest inspection
        - Circular area coverage
        - Artistic/cinematic shots
        
        Args:
            config: Survey configuration
            inward: If True, spiral inward; if False, spiral outward
            
        Returns list of (lat, lon, alt) waypoints.
        """
        waypoints = []
        
        # Calculate spiral parameters
        max_radius = min(config.area_width, config.area_height) / 2
        num_rings = int(max_radius / config.line_spacing)
        points_per_ring = 12  # Points per 360 degrees
        
        radii = list(range(num_rings, 0, -1)) if inward else list(range(1, num_rings + 1))
        
        for ring_idx, ring_num in enumerate(radii):
            radius_m = ring_num * config.line_spacing
            radius_lat = radius_m / cls.METERS_PER_DEG_LAT
            radius_lon = radius_m / cls.METERS_PER_DEG_LON
            
            for point in range(points_per_ring):
                angle = (2 * math.pi * point) / points_per_ring
                # Offset angle for each ring to create spiral effect
                angle += (ring_idx * math.pi / 6)
                
                lat = config.center_lat + (radius_lat * math.cos(angle))
                lon = config.center_lon + (radius_lon * math.sin(angle))
                waypoints.append((lat, lon, config.altitude))
        
        # End at center if spiraling inward
        if inward:
            waypoints.append((config.center_lat, config.center_lon, config.altitude))
        
        return waypoints
    
    @classmethod
    def generate_expanding_square(cls, config: SurveyConfig) -> List[Tuple[float, float, float]]:
        """
        Generate an expanding square search pattern.
        
        This pattern is ideal for:
        - Search and rescue operations
        - Lost object searches
        - Systematic area coverage from a point
        
        Returns list of (lat, lon, alt) waypoints.
        """
        waypoints = []
        
        # Start at center
        current_lat = config.center_lat
        current_lon = config.center_lon
        waypoints.append((current_lat, current_lon, config.altitude))
        
        # Calculate step size in degrees
        step_lat = config.line_spacing / cls.METERS_PER_DEG_LAT
        step_lon = config.line_spacing / cls.METERS_PER_DEG_LON
        
        # Maximum number of expansions
        max_expansions = int(min(config.area_width, config.area_height) / (2 * config.line_spacing))
        
        # Directions: N, E, S, W
        directions = [
            (step_lat, 0),   # North
            (0, step_lon),   # East
            (-step_lat, 0),  # South
            (0, -step_lon),  # West
        ]
        
        for expansion in range(1, max_expansions + 1):
            for dir_idx, (dlat, dlon) in enumerate(directions):
                # Number of steps increases with each expansion
                steps = expansion if dir_idx < 2 else expansion
                if dir_idx >= 2:
                    steps = expansion + 1
                
                for _ in range(steps):
                    current_lat += dlat
                    current_lon += dlon
                    waypoints.append((current_lat, current_lon, config.altitude))
        
        return waypoints


class SurveyMission:
    """Execute a survey mission with the generated pattern."""
    
    def __init__(self, config: SurveyConfig, pattern_type: PatternType):
        self.config = config
        self.pattern_type = pattern_type
        self.waypoints: List[Tuple[float, float, float]] = []
        self._generate_pattern()
    
    def _generate_pattern(self):
        """Generate waypoints based on pattern type."""
        generator = SurveyPatternGenerator
        
        if self.pattern_type == PatternType.GRID:
            self.waypoints = generator.generate_grid_pattern(self.config)
        elif self.pattern_type == PatternType.SPIRAL:
            self.waypoints = generator.generate_spiral_pattern(self.config)
        elif self.pattern_type == PatternType.EXPANDING_SQUARE:
            self.waypoints = generator.generate_expanding_square(self.config)
    
    def get_stats(self) -> dict:
        """Calculate mission statistics."""
        total_distance = 0.0
        for i in range(len(self.waypoints) - 1):
            w1, w2 = self.waypoints[i], self.waypoints[i + 1]
            lat_diff = (w2[0] - w1[0]) * 111000
            lon_diff = (w2[1] - w1[1]) * 85000
            total_distance += math.sqrt(lat_diff**2 + lon_diff**2)
        
        flight_time = total_distance / self.config.speed
        num_photos = int(flight_time / self.config.camera_interval) if self.config.camera_trigger else 0
        
        return {
            "pattern": self.pattern_type.value,
            "waypoints": len(self.waypoints),
            "distance_m": round(total_distance, 1),
            "flight_time_min": round(flight_time / 60, 1),
            "estimated_photos": num_photos,
            "area_coverage_m2": self.config.area_width * self.config.area_height,
        }
    
    async def execute(self):
        """Execute the survey mission (simulation)."""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print(f"Survey Mission: {self.pattern_type.value.upper()} Pattern")
        print(f"{'='*60}")
        print(f"Center: ({self.config.center_lat:.6f}, {self.config.center_lon:.6f})")
        print(f"Area: {self.config.area_width}m x {self.config.area_height}m")
        print(f"Altitude: {self.config.altitude}m")
        print(f"Line Spacing: {self.config.line_spacing}m")
        print(f"Speed: {self.config.speed} m/s")
        print(f"{'='*60}")
        print(f"Waypoints: {stats['waypoints']}")
        print(f"Total Distance: {stats['distance_m']}m")
        print(f"Est. Flight Time: {stats['flight_time_min']} minutes")
        print(f"Est. Photos: {stats['estimated_photos']}")
        print(f"{'='*60}\n")
        
        # Simulate mission execution
        print("Taking off...")
        await asyncio.sleep(0.5)
        print(f"Reached survey altitude: {self.config.altitude}m\n")
        
        print("Beginning survey pattern...")
        for i, (lat, lon, alt) in enumerate(self.waypoints[:5], 1):  # Show first 5
            print(f"  Waypoint {i}: ({lat:.6f}, {lon:.6f}, {alt}m)")
            await asyncio.sleep(0.1)
        
        if len(self.waypoints) > 5:
            print(f"  ... {len(self.waypoints) - 5} more waypoints ...")
        
        print("\nSurvey complete!")
        print("Returning to launch...")
        await asyncio.sleep(0.3)
        print("Landing...")
        print("Mission complete!\n")


async def main():
    """Demonstrate different survey patterns."""
    
    # Example location (San Francisco)
    base_config = SurveyConfig(
        center_lat=37.7749,
        center_lon=-122.4194,
        altitude=30.0,
        area_width=50.0,
        area_height=50.0,
        line_spacing=8.0,
        speed=4.0,
        camera_trigger=True,
        camera_interval=2.0,
    )
    
    print("Drone AI - Survey Pattern Examples")
    print("===================================\n")
    
    # Demo each pattern type
    patterns = [PatternType.GRID, PatternType.SPIRAL, PatternType.EXPANDING_SQUARE]
    
    for pattern in patterns:
        mission = SurveyMission(base_config, pattern)
        await mission.execute()
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
