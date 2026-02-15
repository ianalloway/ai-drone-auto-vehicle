"""
RRT* (Rapidly-exploring Random Trees) Path Planning for Drone AI

Implements RRT* algorithm for path planning in complex environments
with dynamic obstacles.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from loguru import logger


@dataclass
class RRTNode:
    """Node in the RRT tree."""
    position: np.ndarray
    parent: Optional['RRTNode'] = None
    cost: float = 0.0
    children: List['RRTNode'] = field(default_factory=list)


class RRTStarPlanner:
    """
    RRT* (Optimal Rapidly-exploring Random Trees) path planner.
    
    Suitable for high-dimensional spaces and complex obstacle configurations.
    Provides asymptotically optimal paths.
    """
    
    def __init__(
        self,
        bounds: Tuple[Tuple[float, float], ...],
        step_size: float = 5.0,
        goal_sample_rate: float = 0.1,
        max_iterations: int = 5000,
        search_radius: float = 15.0
    ):
        """
        Initialize RRT* planner.
        
        Args:
            bounds: Search space bounds ((x_min, x_max), (y_min, y_max), ...)
            step_size: Maximum step size for tree expansion
            goal_sample_rate: Probability of sampling the goal
            max_iterations: Maximum planning iterations
            search_radius: Radius for rewiring neighbors
        """
        self.bounds = bounds
        self.dim = len(bounds)
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.search_radius = search_radius
        
        self.obstacles: List[Tuple[np.ndarray, float]] = []  # (center, radius)
        self.nodes: List[RRTNode] = []
        
        logger.info(f"RRTStarPlanner initialized with {self.dim}D space")
    
    def set_obstacles(self, obstacles: List[Tuple[Tuple[float, ...], float]]):
        """
        Set circular/spherical obstacles.
        
        Args:
            obstacles: List of (center, radius) tuples
        """
        self.obstacles = [(np.array(c), r) for c, r in obstacles]
        logger.debug(f"Set {len(obstacles)} obstacles")
    
    def plan(
        self,
        start: Tuple[float, ...],
        goal: Tuple[float, ...],
        goal_threshold: float = 5.0
    ) -> Optional[List[Tuple[float, ...]]]:
        """
        Plan path from start to goal using RRT*.
        
        Args:
            start: Starting position
            goal: Goal position
            goal_threshold: Distance threshold to consider goal reached
            
        Returns:
            List of waypoints, or None if no path found
        """
        start_arr = np.array(start)
        goal_arr = np.array(goal)
        
        # Initialize tree with start node
        self.nodes = [RRTNode(position=start_arr)]
        
        best_goal_node = None
        best_cost = float('inf')
        
        for i in range(self.max_iterations):
            # Sample random point (with goal bias)
            if np.random.random() < self.goal_sample_rate:
                sample = goal_arr
            else:
                sample = self._random_sample()
            
            # Find nearest node
            nearest = self._nearest_node(sample)
            
            # Steer towards sample
            new_pos = self._steer(nearest.position, sample)
            
            # Check collision
            if self._collision_free(nearest.position, new_pos):
                # Find nearby nodes for rewiring
                near_nodes = self._near_nodes(new_pos)
                
                # Choose best parent
                min_cost = nearest.cost + self._distance(nearest.position, new_pos)
                best_parent = nearest
                
                for near in near_nodes:
                    cost = near.cost + self._distance(near.position, new_pos)
                    if cost < min_cost and self._collision_free(near.position, new_pos):
                        min_cost = cost
                        best_parent = near
                
                # Create new node
                new_node = RRTNode(
                    position=new_pos,
                    parent=best_parent,
                    cost=min_cost
                )
                best_parent.children.append(new_node)
                self.nodes.append(new_node)
                
                # Rewire nearby nodes
                for near in near_nodes:
                    new_cost = new_node.cost + self._distance(new_node.position, near.position)
                    if new_cost < near.cost and self._collision_free(new_node.position, near.position):
                        # Remove from old parent
                        if near.parent:
                            near.parent.children.remove(near)
                        # Set new parent
                        near.parent = new_node
                        near.cost = new_cost
                        new_node.children.append(near)
                        self._propagate_cost(near)
                
                # Check if goal reached
                dist_to_goal = self._distance(new_pos, goal_arr)
                if dist_to_goal < goal_threshold:
                    total_cost = new_node.cost + dist_to_goal
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_goal_node = new_node
            
            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.debug(f"RRT* iteration {i+1}/{self.max_iterations}")
        
        if best_goal_node is None:
            logger.warning("No path found to goal")
            return None
        
        # Reconstruct path
        path = self._reconstruct_path(best_goal_node)
        path.append(tuple(goal_arr))
        
        logger.info(f"Path found with {len(path)} waypoints, cost: {best_cost:.2f}")
        return path
    
    def _random_sample(self) -> np.ndarray:
        """Generate random sample within bounds."""
        sample = np.zeros(self.dim)
        for i, (low, high) in enumerate(self.bounds):
            sample[i] = np.random.uniform(low, high)
        return sample
    
    def _nearest_node(self, point: np.ndarray) -> RRTNode:
        """Find nearest node in tree to point."""
        min_dist = float('inf')
        nearest = self.nodes[0]
        
        for node in self.nodes:
            dist = self._distance(node.position, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _near_nodes(self, point: np.ndarray) -> List[RRTNode]:
        """Find all nodes within search radius."""
        return [
            node for node in self.nodes
            if self._distance(node.position, point) < self.search_radius
        ]
    
    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Steer from one position towards another with step size limit."""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        
        if dist <= self.step_size:
            return to_pos.copy()
        
        return from_pos + (direction / dist) * self.step_size
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Euclidean distance between two points."""
        return float(np.linalg.norm(p1 - p2))
    
    def _collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if path between two points is collision-free."""
        # Check multiple points along the path
        num_checks = max(int(self._distance(p1, p2) / 1.0), 2)
        
        for i in range(num_checks + 1):
            t = i / num_checks
            point = p1 + t * (p2 - p1)
            
            # Check against all obstacles
            for center, radius in self.obstacles:
                if self._distance(point, center) < radius:
                    return False
        
        return True
    
    def _propagate_cost(self, node: RRTNode):
        """Propagate cost update to children."""
        for child in node.children:
            child.cost = node.cost + self._distance(node.position, child.position)
            self._propagate_cost(child)
    
    def _reconstruct_path(self, node: RRTNode) -> List[Tuple[float, ...]]:
        """Reconstruct path from node to root."""
        path = []
        current = node
        
        while current is not None:
            path.append(tuple(current.position))
            current = current.parent
        
        return list(reversed(path))
