# Drone AI Examples

This directory contains example scripts demonstrating various capabilities of the Drone AI framework.

## Examples

### 1. Waypoint Mission (`waypoint_mission.py`)

A simple waypoint navigation example that demonstrates:
- Creating GPS waypoints
- Configuring speed and hold times
- Return-to-launch functionality
- Mission distance and duration estimation

```bash
python waypoint_mission.py
```

### 2. Survey Patterns (`survey_pattern.py`)

Automated survey pattern generation for mapping and inspection:
- **Grid/Lawnmower Pattern**: Ideal for aerial mapping and photogrammetry
- **Spiral Pattern**: Perfect for point-of-interest inspection
- **Expanding Square**: Best for search and rescue operations

```bash
python survey_pattern.py
```

### 3. Obstacle Avoidance (`obstacle_avoidance.py`)

Demonstrates obstacle detection and avoidance algorithms:
- Simulated sensor data processing
- Threat level evaluation
- Multiple avoidance strategies (stop, go around, go over/under)
- Real-time decision making

```bash
python obstacle_avoidance.py
```

## Running the Examples

All examples are self-contained simulations that don't require actual drone hardware:

```bash
# Run any example
python examples/waypoint_mission.py
python examples/survey_pattern.py
python examples/obstacle_avoidance.py
```

## Adapting for Real Hardware

To use these examples with actual drone hardware:

1. **Install DroneAI framework** (when available)
2. **Configure MAVLink connection** to your flight controller
3. **Replace simulation classes** with actual DroneAI imports
4. **Test in simulation first** (SITL/Gazebo) before real flights

## Safety Notes

- Always test in simulation before flying
- Ensure proper GPS lock before waypoint missions
- Set appropriate geofence boundaries
- Have manual override ready at all times
- Follow local drone regulations

## Contributing

Feel free to add more examples! Good candidates include:
- Follow-me mode
- Precision landing
- Multi-drone coordination
- Computer vision integration
- Autonomous inspection routines
