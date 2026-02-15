"""Tests for planning module."""
import pytest
import numpy as np
from src.planning.astar import AStarPlanner
from src.planning.rrt import RRTStarPlanner
from src.planning.avoidance import ObstacleAvoider


class TestAStarPlanner:
    """Test cases for A* path planner."""

    def test_planner_initialization(self):
        """Test planner initializes correctly."""
        planner = AStarPlanner(grid_size=100, safety_margin=2.0)
        assert planner.grid_size == 100
        assert planner.safety_margin == 2.0

    def test_plan_simple_path(self):
        """Test planning a simple path without obstacles."""
        planner = AStarPlanner(grid_size=50)
        start = (0, 0, 0)
        goal = (10, 10, 0)
        path = planner.plan(start, goal, obstacles=[])
        assert path is not None
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal

    def test_plan_with_obstacles(self):
        """Test planning around obstacles."""
        planner = AStarPlanner(grid_size=50)
        start = (0, 0, 0)
        goal = (10, 0, 0)
        obstacles = [(5, 0, 0, 2)]  # x, y, z, radius
        path = planner.plan(start, goal, obstacles=obstacles)
        assert path is not None
        # Path should avoid the obstacle
        for point in path:
            dist = np.sqrt((point[0] - 5)**2 + (point[1] - 0)**2)
            assert dist >= 1.5 or point == start or point == goal

    def test_path_smoothing(self):
        """Test path smoothing function."""
        planner = AStarPlanner(grid_size=50)
        raw_path = [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0)]
        smoothed = planner.smooth_path(raw_path)
        assert len(smoothed) >= 2


class TestRRTStarPlanner:
    """Test cases for RRT* path planner."""

    def test_planner_initialization(self):
        """Test RRT* planner initializes correctly."""
        planner = RRTStarPlanner(
            bounds=((0, 100), (0, 100), (0, 50)),
            step_size=5.0,
            max_iterations=1000
        )
        assert planner.step_size == 5.0
        assert planner.max_iterations == 1000

    def test_plan_in_open_space(self):
        """Test RRT* planning in open space."""
        planner = RRTStarPlanner(
            bounds=((0, 50), (0, 50), (0, 20)),
            step_size=3.0,
            max_iterations=500
        )
        start = (5, 5, 5)
        goal = (40, 40, 10)
        path = planner.plan(start, goal, obstacles=[])
        assert path is not None
        assert len(path) >= 2

    def test_set_obstacles(self):
        """Test setting obstacles."""
        planner = RRTStarPlanner(bounds=((0, 50), (0, 50), (0, 20)))
        obstacles = [(25, 25, 10, 5)]
        planner.set_obstacles(obstacles)
        assert len(planner.obstacles) == 1


class TestObstacleAvoider:
    """Test cases for obstacle avoidance."""

    def test_avoider_initialization(self):
        """Test avoider initializes correctly."""
        avoider = ObstacleAvoider(safety_distance=3.0, max_speed=10.0)
        assert avoider.safety_distance == 3.0
        assert avoider.max_speed == 10.0

    def test_compute_avoidance_no_obstacles(self):
        """Test avoidance with no obstacles."""
        avoider = ObstacleAvoider()
        position = np.array([0, 0, 10])
        velocity = np.array([5, 0, 0])
        goal = np.array([50, 0, 10])
        
        new_velocity = avoider.compute_avoidance(position, velocity, goal, obstacles=[])
        assert new_velocity is not None
        assert len(new_velocity) == 3

    def test_emergency_stop(self):
        """Test emergency stop capability."""
        avoider = ObstacleAvoider()
        avoider.emergency_stop()
        assert avoider.stopped is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
