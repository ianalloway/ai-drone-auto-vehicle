#!/usr/bin/env python3
"""
Obstacle Avoidance Example
==========================

This example demonstrates basic obstacle detection and avoidance
algorithms for autonomous drone navigation.

Features:
- Simulated sensor data processing
- Multiple avoidance strategies
- Path replanning

Usage:
    python obstacle_avoidance.py

Requirements:
    - DroneAI framework installed
    - Obstacle detection sensors (simulated here)
"""

import asyncio
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class AvoidanceStrategy(Enum):
    """Obstacle avoidance strategies."""
    STOP_AND_HOVER = "stop_and_hover"
    GO_AROUND = "go_around"
    GO_OVER = "go_over"
    GO_UNDER = "go_under"
    REVERSE = "reverse"


@dataclass
class Obstacle:
    """Detected obstacle."""
    distance: float  # meters
    bearing: float  # degrees from drone heading (0 = straight ahead)
    elevation: float  # degrees from horizontal (+ = above, - = below)
    size_estimate: str  # "small", "medium", "large"
    confidence: float  # 0.0 to 1.0


@dataclass
class SensorReading:
    """Simulated sensor reading from obstacle detection system."""
    timestamp: float
    obstacles: List[Obstacle]
    clear_path: bool


class ObstacleDetector:
    """
    Simulated obstacle detection system.
    
    In a real implementation, this would interface with:
    - LiDAR sensors
    - Ultrasonic sensors
    - Stereo cameras
    - Depth cameras (Intel RealSense, etc.)
    """
    
    def __init__(self, detection_range: float = 10.0, fov: float = 120.0):
        self.detection_range = detection_range  # meters
        self.fov = fov  # field of view in degrees
        self.min_detection_distance = 0.5  # meters
    
    def scan(self) -> SensorReading:
        """
        Perform a sensor scan and return detected obstacles.
        
        This is a simulation - real implementation would read from actual sensors.
        """
        obstacles = []
        
        # Simulate random obstacle detection (30% chance)
        if random.random() < 0.3:
            num_obstacles = random.randint(1, 3)
            for _ in range(num_obstacles):
                obstacle = Obstacle(
                    distance=random.uniform(2.0, self.detection_range),
                    bearing=random.uniform(-self.fov/2, self.fov/2),
                    elevation=random.uniform(-30, 30),
                    size_estimate=random.choice(["small", "medium", "large"]),
                    confidence=random.uniform(0.7, 1.0)
                )
                obstacles.append(obstacle)
        
        return SensorReading(
            timestamp=asyncio.get_event_loop().time(),
            obstacles=obstacles,
            clear_path=len(obstacles) == 0
        )


class ObstacleAvoidanceSystem:
    """
    Obstacle avoidance decision-making system.
    
    This system:
    1. Processes sensor data
    2. Evaluates threats
    3. Selects avoidance strategy
    4. Generates avoidance maneuvers
    """
    
    # Distance thresholds (meters)
    CRITICAL_DISTANCE = 2.0  # Emergency stop
    WARNING_DISTANCE = 5.0   # Begin avoidance
    CAUTION_DISTANCE = 8.0   # Slow down
    
    def __init__(self, detector: ObstacleDetector):
        self.detector = detector
        self.current_strategy: Optional[AvoidanceStrategy] = None
        self.avoidance_active = False
    
    def evaluate_threat(self, obstacle: Obstacle) -> str:
        """Evaluate threat level of an obstacle."""
        if obstacle.distance < self.CRITICAL_DISTANCE:
            return "CRITICAL"
        elif obstacle.distance < self.WARNING_DISTANCE:
            return "WARNING"
        elif obstacle.distance < self.CAUTION_DISTANCE:
            return "CAUTION"
        return "NONE"
    
    def select_strategy(self, obstacles: List[Obstacle]) -> AvoidanceStrategy:
        """
        Select the best avoidance strategy based on obstacle configuration.
        
        Decision factors:
        - Obstacle distance and position
        - Available clearance (above, below, sides)
        - Current flight mode and mission requirements
        """
        if not obstacles:
            return None
        
        # Find the most threatening obstacle
        closest = min(obstacles, key=lambda o: o.distance)
        threat_level = self.evaluate_threat(closest)
        
        if threat_level == "CRITICAL":
            # Emergency - stop immediately
            return AvoidanceStrategy.STOP_AND_HOVER
        
        # Analyze obstacle position to determine best avoidance
        if abs(closest.bearing) > 45:
            # Obstacle is to the side - continue with caution
            return None
        
        if closest.elevation > 15:
            # Obstacle is above - go under if possible
            return AvoidanceStrategy.GO_UNDER
        elif closest.elevation < -15:
            # Obstacle is below - go over
            return AvoidanceStrategy.GO_OVER
        else:
            # Obstacle is directly ahead - go around
            return AvoidanceStrategy.GO_AROUND
    
    def generate_avoidance_command(self, strategy: AvoidanceStrategy, 
                                    obstacle: Obstacle) -> dict:
        """Generate specific avoidance maneuver commands."""
        commands = {
            "strategy": strategy.value,
            "original_heading": 0,  # Would be actual heading
            "maneuver": {}
        }
        
        if strategy == AvoidanceStrategy.STOP_AND_HOVER:
            commands["maneuver"] = {
                "action": "emergency_stop",
                "hover_duration": 3.0,
                "then": "reassess"
            }
        
        elif strategy == AvoidanceStrategy.GO_AROUND:
            # Determine which side has more clearance
            side = "right" if obstacle.bearing < 0 else "left"
            commands["maneuver"] = {
                "action": "lateral_offset",
                "direction": side,
                "offset_distance": 5.0,  # meters
                "maintain_altitude": True
            }
        
        elif strategy == AvoidanceStrategy.GO_OVER:
            commands["maneuver"] = {
                "action": "climb",
                "altitude_change": 3.0,  # meters
                "maintain_heading": True
            }
        
        elif strategy == AvoidanceStrategy.GO_UNDER:
            commands["maneuver"] = {
                "action": "descend",
                "altitude_change": -3.0,  # meters
                "maintain_heading": True,
                "min_altitude": 2.0  # Don't go below this
            }
        
        elif strategy == AvoidanceStrategy.REVERSE:
            commands["maneuver"] = {
                "action": "reverse",
                "distance": 3.0,  # meters
                "then": "find_alternate_path"
            }
        
        return commands
    
    async def run_avoidance_loop(self, duration: float = 10.0):
        """
        Main obstacle avoidance loop.
        
        This runs continuously during flight, scanning for obstacles
        and taking evasive action when necessary.
        """
        print("\n" + "="*60)
        print("Obstacle Avoidance System Active")
        print("="*60)
        print(f"Detection Range: {self.detector.detection_range}m")
        print(f"Field of View: {self.detector.fov} degrees")
        print(f"Critical Distance: {self.CRITICAL_DISTANCE}m")
        print(f"Warning Distance: {self.WARNING_DISTANCE}m")
        print("="*60 + "\n")
        
        start_time = asyncio.get_event_loop().time()
        scan_count = 0
        obstacles_detected = 0
        avoidance_maneuvers = 0
        
        while (asyncio.get_event_loop().time() - start_time) < duration:
            # Perform sensor scan
            reading = self.detector.scan()
            scan_count += 1
            
            if reading.obstacles:
                obstacles_detected += len(reading.obstacles)
                print(f"[Scan {scan_count}] Detected {len(reading.obstacles)} obstacle(s):")
                
                for i, obs in enumerate(reading.obstacles, 1):
                    threat = self.evaluate_threat(obs)
                    print(f"  Obstacle {i}: {obs.distance:.1f}m, "
                          f"bearing {obs.bearing:.0f}deg, "
                          f"size={obs.size_estimate}, "
                          f"threat={threat}")
                
                # Select and execute avoidance strategy
                strategy = self.select_strategy(reading.obstacles)
                if strategy:
                    avoidance_maneuvers += 1
                    closest = min(reading.obstacles, key=lambda o: o.distance)
                    commands = self.generate_avoidance_command(strategy, closest)
                    
                    print(f"  -> Strategy: {strategy.value}")
                    print(f"  -> Maneuver: {commands['maneuver']}")
                    self.avoidance_active = True
                else:
                    print("  -> No avoidance needed (obstacle not in path)")
            else:
                if self.avoidance_active:
                    print(f"[Scan {scan_count}] Path clear - resuming normal flight")
                    self.avoidance_active = False
            
            await asyncio.sleep(0.5)  # Scan rate: 2 Hz
        
        # Summary
        print("\n" + "="*60)
        print("Obstacle Avoidance Session Summary")
        print("="*60)
        print(f"Total Scans: {scan_count}")
        print(f"Obstacles Detected: {obstacles_detected}")
        print(f"Avoidance Maneuvers: {avoidance_maneuvers}")
        print("="*60 + "\n")


async def main():
    """Demonstrate obstacle avoidance system."""
    print("Drone AI - Obstacle Avoidance Example")
    print("=====================================\n")
    
    # Create detector and avoidance system
    detector = ObstacleDetector(detection_range=10.0, fov=120.0)
    avoidance_system = ObstacleAvoidanceSystem(detector)
    
    # Run avoidance loop for demonstration
    await avoidance_system.run_avoidance_loop(duration=8.0)


if __name__ == "__main__":
    asyncio.run(main())
