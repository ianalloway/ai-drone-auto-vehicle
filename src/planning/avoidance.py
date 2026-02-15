"""
Obstacle Avoidance Module for Drone AI

Provides real-time obstacle avoidance using velocity obstacles
and potential fields for reactive navigation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from loguru import logger


@dataclass
class Obstacle:
    """Represents a dynamic obstacle."""
    position: np.ndarray
    velocity: np.ndarray
    radius: float


@dataclass
class AvoidanceCommand:
    """Command output from obstacle avoidance."""
    velocity: np.ndarray
    is_safe: bool
    closest_obstacle_dist: float
    avoidance_active: bool


class ObstacleAvoider:
    """
    Real-time obstacle avoidance using velocity obstacles and potential fields.
    
    Combines reactive avoidance with goal-directed navigation for smooth
    obstacle avoidance behavior.
    """
    
    def __init__(
        self,
        max_speed: float = 10.0,
        safety_distance: float = 5.0,
        avoidance_gain: float = 2.0,
        goal_gain: float = 1.0,
        time_horizon: float = 3.0
    ):
        """
        Initialize obstacle avoider.
        
        Args:
            max_speed: Maximum allowed speed
            safety_distance: Minimum distance to maintain from obstacles
            avoidance_gain: Gain for avoidance force
            goal_gain: Gain for goal attraction force
            time_horizon: Time horizon for velocity obstacle calculation
        """
        self.max_speed = max_speed
        self.safety_distance = safety_distance
        self.avoidance_gain = avoidance_gain
        self.goal_gain = goal_gain
        self.time_horizon = time_horizon
        
        logger.info("ObstacleAvoider initialized")
    
    def compute_avoidance(
        self,
        position: np.ndarray,
        current_velocity: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle]
    ) -> AvoidanceCommand:
        """
        Compute avoidance velocity command.
        
        Args:
            position: Current position
            current_velocity: Current velocity
            goal: Goal position
            obstacles: List of obstacles
            
        Returns:
            AvoidanceCommand with safe velocity
        """
        # Calculate goal attraction force
        goal_direction = goal - position
        goal_dist = np.linalg.norm(goal_direction)
        
        if goal_dist > 0:
            goal_force = (goal_direction / goal_dist) * self.goal_gain
        else:
            goal_force = np.zeros_like(position)
        
        # Calculate obstacle repulsion forces
        avoidance_force = np.zeros_like(position)
        closest_dist = float('inf')
        avoidance_active = False
        
        for obs in obstacles:
            # Vector from obstacle to agent
            diff = position - obs.position
            dist = np.linalg.norm(diff) - obs.radius
            
            if dist < closest_dist:
                closest_dist = dist
            
            # Apply repulsion if within safety distance
            if dist < self.safety_distance:
                avoidance_active = True
                
                if dist > 0:
                    # Repulsion force inversely proportional to distance
                    repulsion_mag = self.avoidance_gain * (1.0 / dist - 1.0 / self.safety_distance)
                    repulsion_dir = diff / (np.linalg.norm(diff) + 1e-6)
                    avoidance_force += repulsion_mag * repulsion_dir
                else:
                    # Emergency: inside obstacle
                    avoidance_force += diff * self.avoidance_gain * 10
        
        # Combine forces
        total_force = goal_force + avoidance_force
        
        # Apply velocity obstacle constraints
        desired_velocity = self._apply_velocity_obstacles(
            position, current_velocity, total_force, obstacles
        )
        
        # Limit speed
        speed = np.linalg.norm(desired_velocity)
        if speed > self.max_speed:
            desired_velocity = (desired_velocity / speed) * self.max_speed
        
        # Check if safe
        is_safe = closest_dist > self.safety_distance * 0.5
        
        return AvoidanceCommand(
            velocity=desired_velocity,
            is_safe=is_safe,
            closest_obstacle_dist=closest_dist,
            avoidance_active=avoidance_active
        )
    
    def _apply_velocity_obstacles(
        self,
        position: np.ndarray,
        current_velocity: np.ndarray,
        desired_velocity: np.ndarray,
        obstacles: List[Obstacle]
    ) -> np.ndarray:
        """
        Apply velocity obstacle constraints.
        
        Modifies desired velocity to avoid collision with moving obstacles.
        """
        # Sample velocities around desired
        num_samples = 36
        best_velocity = desired_velocity.copy()
        best_cost = float('inf')
        
        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            
            # Sample velocities at different speeds and angles
            for speed_factor in [0.5, 0.75, 1.0]:
                speed = np.linalg.norm(desired_velocity) * speed_factor
                
                if len(position) == 2:
                    sample_vel = np.array([
                        speed * np.cos(angle),
                        speed * np.sin(angle)
                    ])
                else:
                    # 3D case - simplified
                    sample_vel = desired_velocity * speed_factor
                    rot = np.array([
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]
                    ])
                    sample_vel = rot @ sample_vel
                
                # Check if velocity is safe
                if self._is_velocity_safe(position, sample_vel, obstacles):
                    # Cost: deviation from desired + speed penalty
                    cost = np.linalg.norm(sample_vel - desired_velocity)
                    cost += 0.1 * (self.max_speed - np.linalg.norm(sample_vel))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_velocity = sample_vel
        
        return best_velocity
    
    def _is_velocity_safe(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        obstacles: List[Obstacle]
    ) -> bool:
        """Check if velocity leads to collision within time horizon."""
        for obs in obstacles:
            # Relative position and velocity
            rel_pos = obs.position - position
            rel_vel = obs.velocity - velocity
            
            # Time to closest approach
            rel_speed_sq = np.dot(rel_vel, rel_vel)
            if rel_speed_sq < 1e-6:
                continue
            
            t_closest = -np.dot(rel_pos, rel_vel) / rel_speed_sq
            t_closest = np.clip(t_closest, 0, self.time_horizon)
            
            # Distance at closest approach
            closest_pos = rel_pos + rel_vel * t_closest
            closest_dist = np.linalg.norm(closest_pos)
            
            if closest_dist < obs.radius + self.safety_distance:
                return False
        
        return True
    
    def emergency_stop(self) -> AvoidanceCommand:
        """Generate emergency stop command."""
        return AvoidanceCommand(
            velocity=np.zeros(3),
            is_safe=False,
            closest_obstacle_dist=0.0,
            avoidance_active=True
        )
