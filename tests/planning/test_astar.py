"""
Tests for planning/astar.py

Covers:
- Node dataclass: f_cost, comparison, equality, hashing
- AStarPlanner initialization
- set_obstacles() with safety margin expansion
- set_obstacle_grid() direct assignment
- plan() on known maps: simple path, around wall, diagonal
- Edge cases: start == goal, start blocked, goal blocked
- Edge cases: out-of-bounds start/goal, no path exists
- _heuristic() correctness
- smooth_path() basic behavior
"""
import numpy as np
import pytest

from planning.astar import Node, AStarPlanner


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

class TestNode:
    def test_f_cost_is_g_plus_h(self):
        node = Node(position=(0, 0), g_cost=3.0, h_cost=4.0)
        assert node.f_cost == pytest.approx(7.0)

    def test_f_cost_with_inf_g(self):
        node = Node(position=(0, 0))
        assert node.f_cost == float("inf")

    def test_less_than_by_f_cost(self):
        cheap = Node(position=(0, 0), g_cost=1.0, h_cost=1.0)
        expensive = Node(position=(1, 1), g_cost=5.0, h_cost=5.0)
        assert cheap < expensive

    def test_equality_by_position(self):
        a = Node(position=(2, 3), g_cost=1.0)
        b = Node(position=(2, 3), g_cost=99.0)
        assert a == b

    def test_inequality_different_positions(self):
        a = Node(position=(0, 0))
        b = Node(position=(0, 1))
        assert a != b

    def test_hash_consistent_with_position(self):
        a = Node(position=(5, 7), g_cost=1.0)
        b = Node(position=(5, 7), g_cost=2.0)
        assert hash(a) == hash(b)

    def test_hash_different_positions_usually_different(self):
        a = Node(position=(0, 0))
        b = Node(position=(1, 2))
        # Not guaranteed equal
        assert hash(a) != hash(b)

    def test_parent_default_none(self):
        node = Node(position=(0, 0))
        assert node.parent is None


# ---------------------------------------------------------------------------
# AStarPlanner initialization
# ---------------------------------------------------------------------------

class TestAStarPlannerInit:
    def test_default_grid_size(self):
        planner = AStarPlanner()
        assert planner.grid_size == (100, 100)

    def test_custom_grid_size(self):
        planner = AStarPlanner(grid_size=(20, 30))
        assert planner.grid_size == (20, 30)

    def test_obstacle_grid_starts_empty(self):
        planner = AStarPlanner(grid_size=(10, 10))
        assert planner.obstacle_grid.sum() == 0

    def test_obstacle_grid_dtype(self):
        planner = AStarPlanner()
        assert planner.obstacle_grid.dtype == np.uint8

    def test_diagonal_movement_default_true(self):
        planner = AStarPlanner()
        assert planner.diagonal_movement is True


# ---------------------------------------------------------------------------
# set_obstacles()
# ---------------------------------------------------------------------------

class TestSetObstacles:
    def test_single_obstacle_marks_cell(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        planner.set_obstacles([(5, 5)])
        assert planner.obstacle_grid[5, 5] == 1

    def test_safety_margin_expands_obstacle(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=1)
        planner.set_obstacles([(5, 5)])
        # With margin=1, a 3x3 area around (5,5) should be marked
        assert planner.obstacle_grid[4, 4] == 1
        assert planner.obstacle_grid[4, 5] == 1
        assert planner.obstacle_grid[5, 4] == 1
        assert planner.obstacle_grid[5, 6] == 1
        assert planner.obstacle_grid[6, 6] == 1

    def test_set_obstacles_clears_previous(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        planner.set_obstacles([(1, 1)])
        planner.set_obstacles([(8, 8)])
        assert planner.obstacle_grid[1, 1] == 0
        assert planner.obstacle_grid[8, 8] == 1

    def test_obstacle_at_boundary_doesnt_raise(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=1)
        planner.set_obstacles([(0, 0), (9, 9)])  # corners

    def test_no_obstacles_grid_stays_empty(self):
        planner = AStarPlanner(grid_size=(10, 10))
        planner.set_obstacles([])
        assert planner.obstacle_grid.sum() == 0


# ---------------------------------------------------------------------------
# set_obstacle_grid()
# ---------------------------------------------------------------------------

class TestSetObstacleGrid:
    def test_direct_grid_assignment(self):
        planner = AStarPlanner(grid_size=(5, 5))
        grid = np.zeros((8, 8), dtype=np.uint8)
        grid[3, 3] = 1
        planner.set_obstacle_grid(grid)
        assert planner.grid_size == (8, 8)
        assert planner.obstacle_grid[3, 3] == 1

    def test_grid_is_copied_not_referenced(self):
        planner = AStarPlanner(grid_size=(5, 5))
        grid = np.zeros((5, 5), dtype=np.uint8)
        planner.set_obstacle_grid(grid)
        grid[0, 0] = 1  # mutate original
        assert planner.obstacle_grid[0, 0] == 0


# ---------------------------------------------------------------------------
# plan() – path found cases
# ---------------------------------------------------------------------------

class TestPlanPathFound:
    def test_start_equals_goal(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        path = planner.plan((5, 5), (5, 5))
        assert path is not None
        assert len(path) >= 1
        assert path[0] == (5, 5)
        assert path[-1] == (5, 5)

    def test_adjacent_cells_path_length_two(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        path = planner.plan((0, 0), (0, 1))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (0, 1)

    def test_simple_straight_line(self):
        planner = AStarPlanner(
            grid_size=(10, 10), safety_margin=0, diagonal_movement=False
        )
        path = planner.plan((0, 0), (0, 5))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (0, 5)

    def test_path_around_horizontal_wall(self):
        """A wall across most of the middle; path must go around it."""
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        # Wall at row=5, cols 0-7
        obstacles = [(5, c) for c in range(8)]
        planner.set_obstacles(obstacles)
        path = planner.plan((0, 0), (9, 9))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    def test_path_start_and_goal_on_known_open_grid(self):
        planner = AStarPlanner(grid_size=(20, 20), safety_margin=0)
        path = planner.plan((0, 0), (19, 19))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (19, 19)

    def test_path_no_duplicate_positions(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        path = planner.plan((0, 0), (9, 9))
        assert path is not None
        assert len(path) == len(set(path)), "Path contains duplicate positions"

    def test_path_all_positions_within_bounds(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        path = planner.plan((0, 0), (9, 9))
        assert path is not None
        for r, c in path:
            assert 0 <= r < 10
            assert 0 <= c < 10

    def test_path_only_free_cells(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        planner.set_obstacles([(2, 2), (3, 3)])
        path = planner.plan((0, 0), (9, 9))
        assert path is not None
        for pos in path:
            assert planner.obstacle_grid[pos[0], pos[1]] == 0

    def test_each_step_is_adjacent(self):
        """Consecutive waypoints must differ by at most 1 in each dimension."""
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        path = planner.plan((0, 0), (9, 9))
        assert path is not None
        for i in range(len(path) - 1):
            dr = abs(path[i + 1][0] - path[i][0])
            dc = abs(path[i + 1][1] - path[i][1])
            assert dr <= 1
            assert dc <= 1


# ---------------------------------------------------------------------------
# plan() – no path / invalid cases
# ---------------------------------------------------------------------------

class TestPlanNoPath:
    def test_start_out_of_bounds_returns_none(self):
        planner = AStarPlanner(grid_size=(10, 10))
        path = planner.plan((-1, 0), (5, 5))
        assert path is None

    def test_goal_out_of_bounds_returns_none(self):
        planner = AStarPlanner(grid_size=(10, 10))
        path = planner.plan((0, 0), (15, 15))
        assert path is None

    def test_start_blocked_returns_none(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        planner.set_obstacles([(0, 0)])
        path = planner.plan((0, 0), (9, 9))
        assert path is None

    def test_goal_blocked_returns_none(self):
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        planner.set_obstacles([(9, 9)])
        path = planner.plan((0, 0), (9, 9))
        assert path is None

    def test_completely_walled_goal_returns_none(self):
        """Surround goal with obstacles so there's no path."""
        planner = AStarPlanner(grid_size=(10, 10), safety_margin=0)
        # Fill entire grid with obstacles except (0,0)
        grid = np.ones((10, 10), dtype=np.uint8)
        grid[0, 0] = 0
        planner.set_obstacle_grid(grid)
        path = planner.plan((0, 0), (9, 9))
        assert path is None


# ---------------------------------------------------------------------------
# _heuristic()
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_same_point_heuristic_zero(self):
        planner = AStarPlanner()
        assert planner._heuristic((3, 4), (3, 4)) == pytest.approx(0.0)

    def test_pythagorean_triple(self):
        # distance from (0,0) to (3,4) = 5
        planner = AStarPlanner()
        assert planner._heuristic((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_heuristic_symmetric(self):
        planner = AStarPlanner()
        assert planner._heuristic((1, 2), (4, 6)) == pytest.approx(
            planner._heuristic((4, 6), (1, 2))
        )

    def test_heuristic_nonnegative(self):
        planner = AStarPlanner()
        assert planner._heuristic((0, 0), (10, 10)) >= 0


# ---------------------------------------------------------------------------
# smooth_path()
# ---------------------------------------------------------------------------

class TestSmoothPath:
    def test_path_length_unchanged(self):
        planner = AStarPlanner()
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        smoothed = planner.smooth_path(path)
        assert len(smoothed) == len(path)

    def test_endpoints_preserved(self):
        planner = AStarPlanner()
        path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        smoothed = planner.smooth_path(path, iterations=100)
        # Start and end should stay the same
        assert smoothed[0] == path[0]
        assert smoothed[-1] == path[-1]

    def test_short_path_returned_unchanged(self):
        planner = AStarPlanner()
        path = [(0, 0), (5, 5)]
        smoothed = planner.smooth_path(path)
        assert smoothed == path

    def test_single_element_path_returned_unchanged(self):
        planner = AStarPlanner()
        path = [(3, 3)]
        smoothed = planner.smooth_path(path)
        assert smoothed == path

    def test_smoothed_result_is_list_of_tuples(self):
        planner = AStarPlanner()
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        smoothed = planner.smooth_path(path)
        for point in smoothed:
            assert isinstance(point, tuple)
            assert len(point) == 2
