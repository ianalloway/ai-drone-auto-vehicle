"""
A* Path Planning Algorithm for Drone AI

Implements A* pathfinding for global path planning in known environments.
"""

import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from loguru import logger


@dataclass
class Node:
    """Represents a node in the search graph."""
    position: Tuple[int, int]
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = 0.0  # Heuristic cost to goal
    parent: Optional['Node'] = None
    
    @property
    def f_cost(self) -> float:
        """Total estimated cost."""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost
    
    def __hash__(self) -> int:
        return hash(self.position)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.position == other.position


class AStarPlanner:
    """
    A* pathfinding algorithm for 2D/3D grid-based navigation.
    
    Finds the optimal path from start to goal while avoiding obstacles.
    """
    
    # 8-directional movement (including diagonals)
    DIRECTIONS_2D = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
    ]
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        diagonal_movement: bool = True,
        safety_margin: int = 1
    ):
        """
        Initialize the A* planner.
        
        Args:
            grid_size: Size of the planning grid (rows, cols)
            diagonal_movement: Allow diagonal movement
            safety_margin: Extra cells around obstacles to avoid
        """
        self.grid_size = grid_size
        self.diagonal_movement = diagonal_movement
        self.safety_margin = safety_margin
        
        # Obstacle grid (1 = obstacle, 0 = free)
        self.obstacle_grid = np.zeros(grid_size, dtype=np.uint8)
        
        logger.info(f"AStarPlanner initialized with grid size {grid_size}")
    
    def set_obstacles(self, obstacles: List[Tuple[int, int]]):
        """
        Set obstacle positions on the grid.
        
        Args:
            obstacles: List of (row, col) obstacle positions
        """
        self.obstacle_grid.fill(0)
        
        for obs in obstacles:
            # Add safety margin around obstacles
            for dr in range(-self.safety_margin, self.safety_margin + 1):
                for dc in range(-self.safety_margin, self.safety_margin + 1):
                    r, c = obs[0] + dr, obs[1] + dc
                    if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                        self.obstacle_grid[r, c] = 1
        
        logger.debug(f"Set {len(obstacles)} obstacles with margin {self.safety_margin}")
    
    def set_obstacle_grid(self, grid: np.ndarray):
        """Set the obstacle grid directly."""
        self.obstacle_grid = grid.copy()
        self.grid_size = grid.shape
    
    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find optimal path from start to goal.
        
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            
        Returns:
            List of positions forming the path, or None if no path exists
        """
        if not self._is_valid(start) or not self._is_valid(goal):
            logger.warning("Start or goal position is invalid")
            return None
        
        if self.obstacle_grid[start[0], start[1]] == 1:
            logger.warning("Start position is blocked")
            return None
        
        if self.obstacle_grid[goal[0], goal[1]] == 1:
            logger.warning("Goal position is blocked")
            return None
        
        # Initialize
        start_node = Node(position=start, g_cost=0)
        start_node.h_cost = self._heuristic(start, goal)
        
        open_set: List[Node] = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        node_map = {start: start_node}
        
        iterations = 0
        max_iterations = self.grid_size[0] * self.grid_size[1] * 2
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            
            if current.position == goal:
                path = self._reconstruct_path(current)
                logger.info(f"Path found with {len(path)} waypoints in {iterations} iterations")
                return path
            
            closed_set.add(current.position)
            
            # Explore neighbors
            directions = self.DIRECTIONS_2D if self.diagonal_movement else self.DIRECTIONS_2D[:4]
            
            for dr, dc in directions:
                neighbor_pos = (current.position[0] + dr, current.position[1] + dc)
                
                if not self._is_valid(neighbor_pos):
                    continue
                
                if neighbor_pos in closed_set:
                    continue
                
                if self.obstacle_grid[neighbor_pos[0], neighbor_pos[1]] == 1:
                    continue
                
                # Calculate movement cost (diagonal costs more)
                move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                tentative_g = current.g_cost + move_cost
                
                if neighbor_pos not in node_map:
                    neighbor = Node(position=neighbor_pos)
                    node_map[neighbor_pos] = neighbor
                else:
                    neighbor = node_map[neighbor_pos]
                
                if tentative_g < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self._heuristic(neighbor_pos, goal)
                    
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)
        
        logger.warning(f"No path found after {iterations} iterations")
        return None
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        return (0 <= pos[0] < self.grid_size[0] and 
                0 <= pos[1] < self.grid_size[1])
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate heuristic distance (Euclidean).
        
        Using Euclidean distance for better path quality with diagonal movement.
        """
        return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal node to start."""
        path = []
        current = node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        return list(reversed(path))
    
    def smooth_path(
        self,
        path: List[Tuple[int, int]],
        iterations: int = 50
    ) -> List[Tuple[int, int]]:
        """
        Smooth the path using gradient descent.
        
        Args:
            path: Original path
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        # Convert to float for smoothing
        smooth = [list(p) for p in path]
        
        weight_data = 0.5
        weight_smooth = 0.3
        
        for _ in range(iterations):
            for i in range(1, len(smooth) - 1):
                for j in range(2):
                    original = path[i][j]
                    smooth[i][j] += weight_data * (original - smooth[i][j])
                    smooth[i][j] += weight_smooth * (
                        smooth[i-1][j] + smooth[i+1][j] - 2 * smooth[i][j]
                    )
        
        return [(int(p[0]), int(p[1])) for p in smooth]
