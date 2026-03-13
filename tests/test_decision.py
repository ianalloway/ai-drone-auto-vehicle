"""Tests for decision module: behavior trees and safety monitor."""

import time
import pytest

from src.decision.behavior import (
    NodeStatus,
    ActionNode,
    ConditionNode,
    SequenceNode,
    SelectorNode,
    ParallelNode,
    InverterNode,
    RepeatNode,
    BehaviorTree,
    create_patrol_tree,
)
from src.decision.safety import (
    SafetyLevel,
    SafetyAction,
    SafetyAlert,
    DroneState,
    SafetyMonitor,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _success_action(bb):
    return NodeStatus.SUCCESS


def _failure_action(bb):
    return NodeStatus.FAILURE


def _running_action(bb):
    return NodeStatus.RUNNING


def _make_drone_state(**overrides):
    defaults = dict(
        battery_percent=80.0,
        altitude=50.0,
        ground_speed=5.0,
        vertical_speed=1.0,
        distance_from_home=100.0,
        signal_strength=80.0,
        gps_satellites=10,
        is_armed=True,
        flight_mode="GUIDED",
        heading=0.0,
        pitch=5.0,
        roll=5.0,
    )
    defaults.update(overrides)
    return DroneState(**defaults)


# ─────────────────────────────────────────────────────────────
# ActionNode / ConditionNode
# ─────────────────────────────────────────────────────────────

class TestLeafNodes:
    def test_action_node_success(self):
        node = ActionNode("test", _success_action)
        assert node.tick({}) == NodeStatus.SUCCESS

    def test_action_node_failure(self):
        node = ActionNode("test", _failure_action)
        assert node.tick({}) == NodeStatus.FAILURE

    def test_condition_node_true(self):
        node = ConditionNode("always_true", lambda bb: True)
        assert node.tick({}) == NodeStatus.SUCCESS

    def test_condition_node_false(self):
        node = ConditionNode("always_false", lambda bb: False)
        assert node.tick({}) == NodeStatus.FAILURE

    def test_condition_node_uses_blackboard(self):
        node = ConditionNode("battery_ok", lambda bb: bb.get("battery", 0) > 20)
        assert node.tick({"battery": 50}) == NodeStatus.SUCCESS
        assert node.tick({"battery": 10}) == NodeStatus.FAILURE


# ─────────────────────────────────────────────────────────────
# SequenceNode
# ─────────────────────────────────────────────────────────────

class TestSequenceNode:
    def test_all_success(self):
        seq = SequenceNode("seq", [
            ActionNode("a", _success_action),
            ActionNode("b", _success_action),
        ])
        assert seq.tick({}) == NodeStatus.SUCCESS

    def test_first_failure_stops(self):
        seq = SequenceNode("seq", [
            ActionNode("a", _failure_action),
            ActionNode("b", _success_action),
        ])
        assert seq.tick({}) == NodeStatus.FAILURE

    def test_running_pauses_sequence(self):
        seq = SequenceNode("seq", [
            ActionNode("a", _success_action),
            ActionNode("b", _running_action),
            ActionNode("c", _success_action),
        ])
        assert seq.tick({}) == NodeStatus.RUNNING

    def test_reset_restarts_sequence(self):
        results = [NodeStatus.RUNNING]

        def toggling(bb):
            return results[0]

        seq = SequenceNode("seq", [ActionNode("a", toggling)])
        assert seq.tick({}) == NodeStatus.RUNNING
        results[0] = NodeStatus.SUCCESS
        seq.reset()
        assert seq.tick({}) == NodeStatus.SUCCESS


# ─────────────────────────────────────────────────────────────
# SelectorNode
# ─────────────────────────────────────────────────────────────

class TestSelectorNode:
    def test_first_success_wins(self):
        sel = SelectorNode("sel", [
            ActionNode("a", _success_action),
            ActionNode("b", _failure_action),
        ])
        assert sel.tick({}) == NodeStatus.SUCCESS

    def test_all_fail_returns_failure(self):
        sel = SelectorNode("sel", [
            ActionNode("a", _failure_action),
            ActionNode("b", _failure_action),
        ])
        assert sel.tick({}) == NodeStatus.FAILURE

    def test_running_child_pauses(self):
        sel = SelectorNode("sel", [
            ActionNode("a", _failure_action),
            ActionNode("b", _running_action),
        ])
        assert sel.tick({}) == NodeStatus.RUNNING


# ─────────────────────────────────────────────────────────────
# ParallelNode
# ─────────────────────────────────────────────────────────────

class TestParallelNode:
    def test_all_succeed(self):
        par = ParallelNode("par", [
            ActionNode("a", _success_action),
            ActionNode("b", _success_action),
        ])
        assert par.tick({}) == NodeStatus.SUCCESS

    def test_one_failure_triggers_failure(self):
        par = ParallelNode("par", [
            ActionNode("a", _success_action),
            ActionNode("b", _failure_action),
        ], failure_threshold=1)
        assert par.tick({}) == NodeStatus.FAILURE

    def test_partial_success_is_running(self):
        par = ParallelNode("par", [
            ActionNode("a", _success_action),
            ActionNode("b", _running_action),
        ], success_threshold=2, failure_threshold=2)
        assert par.tick({}) == NodeStatus.RUNNING


# ─────────────────────────────────────────────────────────────
# InverterNode
# ─────────────────────────────────────────────────────────────

class TestInverterNode:
    def test_inverts_success_to_failure(self):
        inv = InverterNode("inv", ActionNode("a", _success_action))
        assert inv.tick({}) == NodeStatus.FAILURE

    def test_inverts_failure_to_success(self):
        inv = InverterNode("inv", ActionNode("a", _failure_action))
        assert inv.tick({}) == NodeStatus.SUCCESS

    def test_running_unchanged(self):
        inv = InverterNode("inv", ActionNode("a", _running_action))
        assert inv.tick({}) == NodeStatus.RUNNING


# ─────────────────────────────────────────────────────────────
# RepeatNode
# ─────────────────────────────────────────────────────────────

class TestRepeatNode:
    def test_repeats_n_times(self):
        counter = [0]

        def inc(bb):
            counter[0] += 1
            return NodeStatus.SUCCESS

        rep = RepeatNode("rep", ActionNode("a", inc), times=3)
        result = rep.tick({})
        assert result == NodeStatus.SUCCESS
        assert counter[0] == 3

    def test_stops_on_failure(self):
        rep = RepeatNode("rep", ActionNode("a", _failure_action), times=5)
        assert rep.tick({}) == NodeStatus.FAILURE

    def test_reset_resets_counter(self):
        counter = [0]

        def inc(bb):
            counter[0] += 1
            return NodeStatus.SUCCESS

        rep = RepeatNode("rep", ActionNode("a", inc), times=2)
        rep.tick({})
        rep.reset()
        rep.tick({})
        assert counter[0] == 4


# ─────────────────────────────────────────────────────────────
# BehaviorTree
# ─────────────────────────────────────────────────────────────

class TestBehaviorTree:
    def test_tick_returns_status(self):
        tree = BehaviorTree(ActionNode("root", _success_action))
        assert tree.tick() == NodeStatus.SUCCESS

    def test_blackboard_set_get(self):
        tree = BehaviorTree(ActionNode("root", _success_action))
        tree.set_blackboard("key", 42)
        assert tree.get_blackboard("key") == 42

    def test_blackboard_default(self):
        tree = BehaviorTree(ActionNode("root", _success_action))
        assert tree.get_blackboard("missing", "default") == "default"

    def test_reset(self):
        tree = BehaviorTree(ActionNode("root", _running_action))
        tree.tick()
        tree.reset()
        assert tree.root.status == NodeStatus.RUNNING

    def test_create_patrol_tree(self):
        waypoints = [(0, 0, 10), (10, 0, 10), (10, 10, 10)]
        tree = create_patrol_tree(waypoints)
        assert isinstance(tree, BehaviorTree)
        tree.set_blackboard("battery_level", 80)
        tree.set_blackboard("signal_strength", 60)
        result = tree.tick()
        assert result in (NodeStatus.SUCCESS, NodeStatus.FAILURE, NodeStatus.RUNNING)

    def test_patrol_tree_low_battery_returns_home(self):
        waypoints = [(0, 0, 10)]
        tree = create_patrol_tree(waypoints)
        tree.set_blackboard("battery_level", 5)
        tree.set_blackboard("signal_strength", 80)
        result = tree.tick()
        # Battery too low → safety_check fails → return_home fires → SUCCESS
        assert result == NodeStatus.SUCCESS
        assert "target_position" in tree.blackboard


# ─────────────────────────────────────────────────────────────
# SafetyMonitor
# ─────────────────────────────────────────────────────────────

class TestSafetyMonitor:
    def test_normal_state_no_action(self):
        monitor = SafetyMonitor()
        state = _make_drone_state()
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE

    def test_battery_warning(self):
        monitor = SafetyMonitor(battery_warning=30.0, battery_critical=15.0)
        state = _make_drone_state(battery_percent=25.0)
        monitor.check_state(state)
        sources = [a.source for a in monitor.get_alerts()]
        assert "battery" in sources

    def test_battery_critical_returns_home(self):
        monitor = SafetyMonitor(battery_critical=15.0, battery_emergency=10.0)
        state = _make_drone_state(battery_percent=12.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME

    def test_battery_emergency_land(self):
        monitor = SafetyMonitor(battery_emergency=10.0)
        state = _make_drone_state(battery_percent=5.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.EMERGENCY_LAND

    def test_altitude_exceeded(self):
        monitor = SafetyMonitor(max_altitude=120.0)
        state = _make_drone_state(altitude=130.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.SLOW_DOWN

    def test_geofence_breach(self):
        monitor = SafetyMonitor(max_distance=500.0)
        monitor.set_geofence(center=(0.0, 0.0), radius=100.0)
        state = _make_drone_state(distance_from_home=150.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME

    def test_geofence_disabled(self):
        monitor = SafetyMonitor(max_distance=500.0)
        monitor.set_geofence(center=(0.0, 0.0), radius=100.0)
        monitor.disable_geofence()
        state = _make_drone_state(distance_from_home=150.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.CONTINUE

    def test_low_gps_hover(self):
        monitor = SafetyMonitor(min_satellites=6)
        state = _make_drone_state(gps_satellites=3)
        action = monitor.check_state(state)
        assert action == SafetyAction.HOVER

    def test_signal_critical_return_home(self):
        monitor = SafetyMonitor(signal_critical=20.0)
        state = _make_drone_state(signal_strength=10.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.RETURN_HOME

    def test_signal_warning_slow_down(self):
        monitor = SafetyMonitor(signal_warning=40.0, signal_critical=20.0)
        state = _make_drone_state(signal_strength=30.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.SLOW_DOWN

    def test_excessive_tilt_hover(self):
        monitor = SafetyMonitor(max_tilt=45.0)
        state = _make_drone_state(pitch=50.0, roll=0.0)
        action = monitor.check_state(state)
        assert action == SafetyAction.HOVER

    def test_highest_severity_wins(self):
        # Battery emergency (EMERGENCY_LAND) > GPS hover (HOVER)
        monitor = SafetyMonitor(
            battery_emergency=10.0,
            min_satellites=6,
        )
        state = _make_drone_state(battery_percent=5.0, gps_satellites=3)
        action = monitor.check_state(state)
        assert action == SafetyAction.EMERGENCY_LAND

    def test_callback_triggered(self):
        received = []
        monitor = SafetyMonitor(battery_warning=30.0)
        monitor.register_callback(received.append)
        state = _make_drone_state(battery_percent=25.0)
        monitor.check_state(state)
        assert len(received) >= 1

    def test_alerts_cleared_on_new_check(self):
        monitor = SafetyMonitor(battery_warning=30.0)
        state_low = _make_drone_state(battery_percent=25.0)
        monitor.check_state(state_low)
        state_ok = _make_drone_state(battery_percent=80.0)
        monitor.check_state(state_ok)
        assert len(monitor.get_alerts()) == 0

    def test_current_level_normal_when_no_alerts(self):
        monitor = SafetyMonitor()
        state = _make_drone_state()
        monitor.check_state(state)
        assert monitor.current_level == SafetyLevel.NORMAL

    def test_safety_action_ordering(self):
        assert SafetyAction.CONTINUE.value < SafetyAction.SLOW_DOWN.value
        assert SafetyAction.SLOW_DOWN.value < SafetyAction.HOVER.value
        assert SafetyAction.HOVER.value < SafetyAction.RETURN_HOME.value
        assert SafetyAction.RETURN_HOME.value < SafetyAction.EMERGENCY_LAND.value
        assert SafetyAction.EMERGENCY_LAND.value < SafetyAction.KILL_MOTORS.value
