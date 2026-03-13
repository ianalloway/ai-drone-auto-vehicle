"""Tests for planning module: A*, RRT*, and obstacle avoidance."""

import numpy as np
import pytest

from src.planning.astar import AStarPlanner, Node
from src.planning.rrt import RRTStarPlanner
from src.planning.avoidance import ObstacleAvoider, Obstacle, AvoidanceCommand


# ─────────────────────────────────────────────────────────────
# AStarPlanner
# ─────────────────────────────────────────────────────────────

class TestAStarPlanner:
    def test_init_defaults(self):
        planner = AStarPlanner()
        assert planner.grid_size == (100, 100)
        assert planner.diagonal_movement is True

    def test_plan_straight_path(self):
        planner = AStarPlanner(grid_size=(10, 10))
        path = planner.plan((0, 0), (0, 9))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (0, 9)

    def test_plan_diagonal_path(self):
        planner = AStarPlanner(grid_size=(10, 10))
        path = planner.plan((0, 0), (9, 9))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    def test_plan_blocked_goal_returns_none(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        planner.set_obstacles([(5, 5)])
        path = planner.plan((0, 0), (5, 5))
        assert path is None

    def test_plan_blocked_start_returns_none(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        planner.set_obstacles([(0, 0)])
        path = planner.plan((0, 0), (9, 9))
        assert path is None

    def test_plan_no_path_when_fully_blocked(self):
        planner = AStarPlanner(grid_size=(5, 5), safety_margin=0)
        # Block a vertical wall across column 2
        planner.set_obstacles([(r, 2) for r in range(5)])
        path = planner.plan((0, 0), (0, 4))
        assert path is None

    def test_plan_invalid_position(self):
        planner = AStarPlanner(grid_size=(10, 10))
        path = planner.plan((-1, 0), (5, 5))
        assert path is None

    def test_path_does_not_pass_through_obstacles(self):
        planner = AStarPlanner(grid_size=(20, 20), safety_margin=0)
        obstacles = [(10, c) for c in range(1, 19)]
        planner.set_obstacles(obstacles)
        # Gap at column 0 – path must go around via (10,0)
        path = planner.plan((0, 0), (19, 10))
        if path is not None:
            for pos in path:
                assert planner.obstacle_grid[pos[0], pos[1]] == 0

    def test_set_obstacles_adds_safety_margin(self):
        planner = AStarPlanner(grid_size=(20, 20), safety_margin=1)
        planner.set_obstacles([(10, 10)])
        # Adjacent cell should also be marked
        assert planner.obstacle_grid[10, 11] == 1

    def test_set_obstacle_grid_directly(self):
        planner = AStarPlanner(grid_size=(10, 10))
        grid = np.zeros((15, 15), dtype=np.uint8)
        grid[7, 7] = 1
        planner.set_obstacle_grid(grid)
        assert planner.grid_size == (15, 15)
        assert planner.obstacle_grid[7, 7] == 1

    def test_smooth_path_maintains_endpoints(self):
        planner = AStarPlanner(grid_size=(50, 50))
        path = planner.plan((0, 0), (49, 49))
        assert path is not None
        smoothed = planner.smooth_path(path)
        assert smoothed[0] == path[0]
        assert smoothed[-1] == path[-1]

    def test_smooth_path_short_returns_unchanged(self):
        planner = AStarPlanner()
        path = [(0, 0), (1, 1)]
        assert planner.smooth_path(path) == path

    def test_heuristic_zero_at_goal(self):
        planner = AStarPlanner()
        assert planner._heuristic((5, 5), (5, 5)) == pytest.approx(0.0)

    def test_heuristic_euclidean(self):
        planner = AStarPlanner()
        assert planner._heuristic((0, 0), (3, 4)) == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────
# Node dataclass
# ─────────────────────────────────────────────────────────────

class TestNode:
    def test_f_cost(self):
        node = Node(position=(0, 0), g_cost=3.0, h_cost=4.0)
        assert node.f_cost == pytest.approx(7.0)

    def test_equality_by_position(self):
        n1 = Node(position=(1, 2))
        n2 = Node(position=(1, 2))
        assert n1 == n2

    def test_hash_by_position(self):
        n1 = Node(position=(1, 2))
        n2 = Node(position=(1, 2))
        assert hash(n1) == hash(n2)

    def test_less_than_uses_f_cost(self):
        cheap = Node(position=(0, 0), g_cost=1.0, h_cost=1.0)
        expensive = Node(position=(1, 1), g_cost=5.0, h_cost=5.0)
        assert cheap < expensive


# ─────────────────────────────────────────────────────────────
# RRTStarPlanner
# ─────────────────────────────────────────────────────────────

class TestRRTStarPlanner:
    def _make_2d_planner(self, **kwargs):
        bounds = ((0.0, 100.0), (0.0, 100.0))
        defaults = dict(step_size=5.0, goal_sample_rate=0.2, max_iterations=2000)
        defaults.update(kwargs)
        return RRTStarPlanner(bounds=bounds, **defaults)

    def test_init(self):
        planner = self._make_2d_planner()
        assert planner.dim == 2
        assert len(planner.nodes) == 0

    def test_plan_returns_path(self):
        np.random.seed(42)
        planner = self._make_2d_planner()
        path = planner.plan((5.0, 5.0), (95.0, 95.0), goal_threshold=8.0)
        assert path is not None
        assert len(path) >= 2

    def test_plan_start_is_first_node(self):
        np.random.seed(0)
        planner = self._make_2d_planner()
        path = planner.plan((10.0, 10.0), (90.0, 90.0), goal_threshold=8.0)
        if path is not None:
            assert path[0] == pytest.approx((10.0, 10.0))

    def test_plan_avoids_obstacle(self):
        np.random.seed(7)
        bounds = ((0.0, 50.0), (0.0, 50.0))
        planner = RRTStarPlanner(
            bounds=bounds,
            step_size=3.0,
            goal_sample_rate=0.2,
            max_iterations=3000,
            search_radius=10.0,
        )
        planner.set_obstacles([((25.0, 25.0), 5.0)])
        path = planner.plan((2.0, 2.0), (48.0, 48.0), goal_threshold=5.0)
        if path is not None:
            for pt in path:
                dist = np.linalg.norm(np.array(pt) - np.array([25.0, 25.0]))
                assert dist >= 5.0

    def test_plan_unreachable_returns_none(self):
        np.random.seed(1)
        bounds = ((0.0, 10.0), (0.0, 10.0))
        planner = RRTStarPlanner(
            bounds=bounds,
            step_size=1.0,
            goal_sample_rate=0.2,
            max_iterations=200,
        )
        # Surround the goal with obstacles so it cannot be reached
        obstacles = [((8.0 + dx * 2, 8.0 + dy * 2), 3.0)
                     for dx in range(-1, 2) for dy in range(-1, 2)]
        planner.set_obstacles(obstacles)
        path = planner.plan((0.0, 0.0), (9.0, 9.0), goal_threshold=0.5)
        assert path is None

    def test_set_obstacles(self):
        planner = self._make_2d_planner()
        planner.set_obstacles([((10.0, 10.0), 5.0)])
        assert len(planner.obstacles) == 1

    def test_collision_free_clear_path(self):
        planner = self._make_2d_planner()
        p1 = np.array([0.0, 0.0])
        p2 = np.array([10.0, 0.0])
        assert planner._collision_free(p1, p2)

    def test_collision_free_blocked_path(self):
        planner = self._make_2d_planner()
        planner.set_obstacles([((5.0, 0.0), 2.0)])
        p1 = np.array([0.0, 0.0])
        p2 = np.array([10.0, 0.0])
        assert not planner._collision_free(p1, p2)

    def test_distance(self):
        planner = self._make_2d_planner()
        d = planner._distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        assert d == pytest.approx(5.0)

    def test_3d_planning(self):
        np.random.seed(99)
        bounds = ((0.0, 50.0), (0.0, 50.0), (0.0, 50.0))
        planner = RRTStarPlanner(
            bounds=bounds,
            step_size=5.0,
            goal_sample_rate=0.3,
            max_iterations=2000,
        )
        path = planner.plan((2.0, 2.0, 2.0), (45.0, 45.0, 45.0), goal_threshold=8.0)
        assert path is not None


# ─────────────────────────────────────────────────────────────
# ObstacleAvoider
# ─────────────────────────────────────────────────────────────

class TestObstacleAvoider:
    def test_init_defaults(self):
        avoider = ObstacleAvoider()
        assert avoider.max_speed > 0

    def test_compute_avoidance_no_obstacles(self):
        avoider = ObstacleAvoider()
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([1.0, 0.0, 0.0])
        goal = np.array([10.0, 0.0, 0.0])
        result = avoider.compute_avoidance(position, velocity, goal, obstacles=[])
        assert isinstance(result, AvoidanceCommand)
        assert result.velocity.shape == (3,)
        assert result.avoidance_active is False

    def test_compute_avoidance_returns_command(self):
        avoider = ObstacleAvoider()
        position = np.array([0.0, 0.0, 5.0])
        velocity = np.array([2.0, 0.0, 0.0])
        goal = np.array([20.0, 0.0, 5.0])
        obs = Obstacle(
            position=np.array([10.0, 0.0, 5.0]),
            velocity=np.zeros(3),
            radius=2.0,
        )
        result = avoider.compute_avoidance(position, velocity, goal, [obs])
        assert isinstance(result, AvoidanceCommand)
        assert result.velocity.shape == (3,)

    def test_avoidance_speed_within_max(self):
        avoider = ObstacleAvoider(max_speed=5.0)
        position = np.zeros(3)
        velocity = np.array([1.0, 0.0, 0.0])
        goal = np.array([50.0, 0.0, 0.0])
        obs = Obstacle(
            position=np.array([5.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            radius=1.5,
        )
        result = avoider.compute_avoidance(position, velocity, goal, [obs])
        assert np.linalg.norm(result.velocity) <= avoider.max_speed + 1e-6

    def test_avoidance_active_when_close(self):
        avoider = ObstacleAvoider(safety_distance=5.0)
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([1.0, 0.0, 0.0])
        goal = np.array([20.0, 0.0, 0.0])
        # Obstacle very close
        obs = Obstacle(
            position=np.array([3.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            radius=1.0,
        )
        result = avoider.compute_avoidance(position, velocity, goal, [obs])
        assert result.avoidance_active is True

    def test_emergency_stop(self):
        avoider = ObstacleAvoider()
        cmd = avoider.emergency_stop()
        assert isinstance(cmd, AvoidanceCommand)
        np.testing.assert_array_equal(cmd.velocity, np.zeros(3))
        assert cmd.is_safe is False

    def test_is_velocity_safe_no_obstacles(self):
        avoider = ObstacleAvoider()
        position = np.zeros(3)
        velocity = np.array([1.0, 0.0, 0.0])
        assert avoider._is_velocity_safe(position, velocity, []) is True
