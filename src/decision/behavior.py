"""
Behavior Tree Module for Drone AI

Implements behavior trees for mission execution and decision making.
"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any, Dict
from loguru import logger


class NodeStatus(Enum):
    """Status of behavior tree node execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorNode(ABC):
    """Abstract base class for behavior tree nodes."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = NodeStatus.RUNNING
    
    @abstractmethod
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Execute one tick of the node."""
        pass
    
    def reset(self):
        """Reset node state."""
        self.status = NodeStatus.RUNNING


class ActionNode(BehaviorNode):
    """Leaf node that performs an action."""
    
    def __init__(self, name: str, action: Callable[[Dict[str, Any]], NodeStatus]):
        super().__init__(name)
        self.action = action
    
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        self.status = self.action(blackboard)
        return self.status


class ConditionNode(BehaviorNode):
    """Leaf node that checks a condition."""
    
    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool]):
        super().__init__(name)
        self.condition = condition
    
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        result = self.condition(blackboard)
        self.status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        return self.status


class SequenceNode(BehaviorNode):
    """Composite node that runs children in sequence until one fails."""
    
    def __init__(self, name: str, children: List[BehaviorNode]):
        super().__init__(name)
        self.children = children
        self.current_child = 0
    
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        while self.current_child < len(self.children):
            child = self.children[self.current_child]
            status = child.tick(blackboard)
            
            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status
            elif status == NodeStatus.FAILURE:
                self.status = NodeStatus.FAILURE
                return self.status
            
            self.current_child += 1
        
        self.status = NodeStatus.SUCCESS
        return self.status
    
    def reset(self):
        super().reset()
        self.current_child = 0
        for child in self.children:
            child.reset()


class SelectorNode(BehaviorNode):
    """Composite node that runs children until one succeeds."""
    
    def __init__(self, name: str, children: List[BehaviorNode]):
        super().__init__(name)
        self.children = children
        self.current_child = 0
    
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        while self.current_child < len(self.children):
            child = self.children[self.current_child]
            status = child.tick(blackboard)
            
            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status
            elif status == NodeStatus.SUCCESS:
                self.status = NodeStatus.SUCCESS
                return self.status
            
            self.current_child += 1
        
        self.status = NodeStatus.FAILURE
        return self.status
    
    def reset(self):
        super().reset()
        self.current_child = 0
        for child in self.children:
            child.reset()


class ParallelNode(BehaviorNode):
    """Composite node that runs all children simultaneously."""
    
    def __init__(
        self, 
        name: str, 
        children: List[BehaviorNode],
        success_threshold: int = None,
        failure_threshold: int = 1
    ):
        super().__init__(name)
        self.children = children
        self.success_threshold = success_threshold or len(children)
        self.failure_threshold = failure_threshold
    
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        success_count = 0
        failure_count = 0
        
        for child in self.children:
            status = child.tick(blackboard)
            
            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1
        
        if failure_count >= self.failure_threshold:
            self.status = NodeStatus.FAILURE
        elif success_count >= self.success_threshold:
            self.status = NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.RUNNING
        
        return self.status
    
    def reset(self):
        super().reset()
        for child in self.children:
            child.reset()


class DecoratorNode(BehaviorNode):
    """Base class for decorator nodes that modify child behavior."""
    
    def __init__(self, name: str, child: BehaviorNode):
        super().__init__(name)
        self.child = child
    
    def reset(self):
        super().reset()
        self.child.reset()


class InverterNode(DecoratorNode):
    """Inverts the result of its child."""
    
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        status = self.child.tick(blackboard)
        
        if status == NodeStatus.SUCCESS:
            self.status = NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.RUNNING
        
        return self.status


class RepeatNode(DecoratorNode):
    """Repeats child execution a specified number of times."""
    
    def __init__(self, name: str, child: BehaviorNode, times: int):
        super().__init__(name, child)
        self.times = times
        self.count = 0
    
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        while self.count < self.times:
            status = self.child.tick(blackboard)
            
            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status
            elif status == NodeStatus.FAILURE:
                self.status = NodeStatus.FAILURE
                return self.status
            
            self.child.reset()
            self.count += 1
        
        self.status = NodeStatus.SUCCESS
        return self.status
    
    def reset(self):
        super().reset()
        self.count = 0


class BehaviorTree:
    """
    Behavior tree for autonomous decision making.
    
    Provides a hierarchical structure for organizing complex behaviors
    into manageable, reusable components.
    """
    
    def __init__(self, root: BehaviorNode):
        self.root = root
        self.blackboard: Dict[str, Any] = {}
        
        logger.info(f"BehaviorTree created with root: {root.name}")
    
    def tick(self) -> NodeStatus:
        """Execute one tick of the behavior tree."""
        return self.root.tick(self.blackboard)
    
    def reset(self):
        """Reset the entire tree."""
        self.root.reset()
    
    def set_blackboard(self, key: str, value: Any):
        """Set a value in the blackboard."""
        self.blackboard[key] = value
    
    def get_blackboard(self, key: str, default: Any = None) -> Any:
        """Get a value from the blackboard."""
        return self.blackboard.get(key, default)


# Pre-built behavior trees for common drone missions

def create_patrol_tree(waypoints: List[tuple]) -> BehaviorTree:
    """Create a behavior tree for patrol mission."""
    
    def check_battery(bb: Dict) -> bool:
        return bb.get("battery_level", 100) > 20
    
    def check_signal(bb: Dict) -> bool:
        return bb.get("signal_strength", 100) > 30
    
    def goto_waypoint(bb: Dict) -> NodeStatus:
        current_wp = bb.get("current_waypoint", 0)
        if current_wp >= len(waypoints):
            bb["current_waypoint"] = 0
            return NodeStatus.SUCCESS
        
        # Simulate movement
        bb["target_position"] = waypoints[current_wp]
        bb["current_waypoint"] = current_wp + 1
        return NodeStatus.SUCCESS
    
    def return_home(bb: Dict) -> NodeStatus:
        bb["target_position"] = bb.get("home_position", (0, 0, 0))
        return NodeStatus.SUCCESS
    
    # Build tree
    safety_check = SequenceNode("safety_check", [
        ConditionNode("battery_ok", check_battery),
        ConditionNode("signal_ok", check_signal)
    ])
    
    patrol_action = ActionNode("goto_waypoint", goto_waypoint)
    return_action = ActionNode("return_home", return_home)
    
    root = SelectorNode("patrol_root", [
        SequenceNode("normal_patrol", [safety_check, patrol_action]),
        return_action
    ])
    
    tree = BehaviorTree(root)
    tree.set_blackboard("current_waypoint", 0)
    
    return tree
