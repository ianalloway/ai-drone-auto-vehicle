# Hardware Compatibility Guide

This document outlines compatible hardware for the Drone AI framework.

## Flight Controllers

### Fully Supported

| Controller | Protocol | Notes |
|------------|----------|-------|
| Pixhawk 4 | MAVLink 2.0 | Recommended for most applications |
| Pixhawk 6X | MAVLink 2.0 | Latest generation, best performance |
| Pixhawk 6C | MAVLink 2.0 | Compact version of 6X |
| Cube Orange+ | MAVLink 2.0 | Industrial-grade reliability |
| Cube Black | MAVLink 2.0 | Legacy support |
| Holybro Kakute H7 | MAVLink 2.0 | Lightweight option |

### Experimental Support

| Controller | Protocol | Notes |
|------------|----------|-------|
| Betaflight FC | MSP | Limited autonomous features |
| DJI N3 | DJI SDK | Requires DJI SDK integration |
| PX4 Mini | MAVLink 2.0 | Size-constrained applications |

## Companion Computers

### Recommended

| Computer | CPU | RAM | Notes |
|----------|-----|-----|-------|
| NVIDIA Jetson Orin Nano | 6-core ARM | 8GB | Best for computer vision |
| NVIDIA Jetson Xavier NX | 6-core ARM | 8GB | Good balance of power/size |
| Raspberry Pi 5 | 4-core ARM | 8GB | Budget-friendly option |
| Intel NUC 12 | i5/i7 | 16GB+ | Maximum processing power |

### Minimum Requirements

- **CPU**: ARM Cortex-A72 or equivalent (4 cores recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 32GB minimum
- **OS**: Ubuntu 20.04+ or Raspberry Pi OS

## Sensors

### Obstacle Detection

| Sensor | Type | Range | Interface |
|--------|------|-------|-----------|
| Intel RealSense D435i | Stereo Depth | 0.2-10m | USB 3.0 |
| Intel RealSense D455 | Stereo Depth | 0.4-20m | USB 3.0 |
| Livox Mid-360 | LiDAR | 0.1-40m | Ethernet |
| TFmini Plus | LiDAR | 0.1-12m | UART/I2C |
| Benewake TF-Luna | LiDAR | 0.2-8m | UART/I2C |
| HC-SR04 | Ultrasonic | 0.02-4m | GPIO |
| MaxBotix MB1240 | Ultrasonic | 0.2-7.6m | Analog/UART |

### GPS/GNSS

| Module | Constellations | RTK | Notes |
|--------|----------------|-----|-------|
| Here3+ | GPS/GLONASS/BeiDou/Galileo | Yes | CAN bus, recommended |
| Here4 | GPS/GLONASS/BeiDou/Galileo | Yes | Latest generation |
| u-blox ZED-F9P | GPS/GLONASS/BeiDou/Galileo | Yes | cm-level accuracy |
| u-blox M8N | GPS/GLONASS | No | Budget option |
| Holybro M9N | GPS/GLONASS/BeiDou/Galileo | No | Good value |

### Cameras

| Camera | Resolution | FPS | Interface | Use Case |
|--------|------------|-----|-----------|----------|
| Raspberry Pi Camera v3 | 12MP | 120fps | CSI | General purpose |
| Arducam IMX477 | 12.3MP | 60fps | CSI | High quality |
| FLIR Blackfly S | 5MP | 75fps | USB 3.0 | Machine vision |
| Basler ace 2 | 5MP | 113fps | USB 3.0 | Industrial |
| Thermal: FLIR Lepton 3.5 | 160x120 | 9fps | SPI | Thermal imaging |

### IMU/Compass

| Sensor | Type | Interface | Notes |
|--------|------|-----------|-------|
| ICM-42688-P | 6-axis IMU | SPI | High performance |
| BMI088 | 6-axis IMU | SPI | Automotive grade |
| IST8310 | Magnetometer | I2C | Built into many FCs |
| RM3100 | Magnetometer | SPI/I2C | High precision |

## Communication

### Telemetry Radios

| Radio | Range | Frequency | Notes |
|-------|-------|-----------|-------|
| Holybro SiK 915MHz | 1-2km | 915MHz | US/Australia |
| Holybro SiK 433MHz | 1-2km | 433MHz | EU/Asia |
| RFD900x | 40km+ | 900MHz | Long range |
| Microhard pDDL | 100km+ | Various | Professional |

### Video Transmission

| System | Range | Latency | Notes |
|--------|-------|---------|-------|
| DJI O3 Air Unit | 10km | 28ms | Best consumer option |
| Herelink | 20km | 110ms | Professional |
| Analog 5.8GHz | 1-5km | <10ms | Low latency |

## Frame Recommendations

### By Application

**Mapping/Survey**
- Frame: Quadcopter or hexacopter
- Size: 450-650mm
- Flight time: 25-45 minutes
- Payload: 500g-2kg

**Inspection**
- Frame: Quadcopter
- Size: 250-450mm
- Flight time: 15-25 minutes
- Payload: 200-500g

**Delivery/Heavy Lift**
- Frame: Hexacopter or octocopter
- Size: 650mm+
- Flight time: 15-30 minutes
- Payload: 2-10kg

**Racing/Agile**
- Frame: Quadcopter
- Size: 180-250mm
- Flight time: 5-10 minutes
- Payload: Minimal

## Wiring Diagrams

### Basic Setup

```
┌─────────────────┐
│  Companion      │
│  Computer       │
│  (Jetson/Pi)    │
└────────┬────────┘
         │ UART/USB
         ▼
┌─────────────────┐     ┌─────────────┐
│  Flight         │────▶│  ESCs       │
│  Controller     │     └─────────────┘
│  (Pixhawk)      │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│  GPS  │ │ Radio │
└───────┘ └───────┘
```

### Advanced Setup with Sensors

```
┌─────────────────────────────────────────────────────┐
│                 Companion Computer                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │ Camera  │  │ Depth   │  │ LiDAR   │             │
│  │ (CSI)   │  │ (USB)   │  │ (ETH)   │             │
│  └─────────┘  └─────────┘  └─────────┘             │
└────────────────────┬────────────────────────────────┘
                     │ UART (MAVLink)
                     ▼
┌─────────────────────────────────────────────────────┐
│                Flight Controller                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│  │GPS  │ │IMU  │ │Baro │ │Mag  │ │Radio│          │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘          │
└────────────────────┬────────────────────────────────┘
                     │ PWM/DShot
                     ▼
              ┌──────────────┐
              │  ESCs/Motors │
              └──────────────┘
```

## Software Requirements

### Operating System
- Ubuntu 20.04 LTS or 22.04 LTS (recommended)
- Raspberry Pi OS (64-bit)

### Dependencies
- Python 3.8+
- MAVLink/pymavlink
- OpenCV 4.x (for vision)
- ROS 2 Humble (optional)

### Firmware
- ArduPilot 4.3+ (recommended)
- PX4 1.13+ (supported)

## Tested Configurations

### Budget Build (~$500)
- Pixhawk 4 Mini
- Raspberry Pi 4 (4GB)
- u-blox M8N GPS
- SiK 915MHz radio
- Generic 450mm frame

### Mid-Range Build (~$1500)
- Pixhawk 6C
- Jetson Orin Nano
- Here3+ GPS (RTK capable)
- Intel RealSense D435i
- Holybro X500 frame

### Professional Build (~$5000+)
- Cube Orange+
- Jetson Xavier NX
- Here4 GPS with RTK base
- Livox Mid-360 LiDAR
- Intel RealSense D455
- Custom carbon fiber frame

## Troubleshooting

### Common Issues

**No MAVLink connection**
- Check baud rate (default: 57600 or 921600)
- Verify UART wiring (TX→RX, RX→TX)
- Ensure flight controller is powered

**GPS not acquiring fix**
- Move to open sky area
- Check antenna orientation
- Wait 2-5 minutes for cold start

**Camera not detected**
- Check CSI ribbon cable connection
- Enable camera in raspi-config
- Verify camera module compatibility

**High CPU usage**
- Reduce camera resolution/FPS
- Disable unused sensors
- Use hardware acceleration (GPU)

## Contributing

Found hardware that works well? Submit a PR to add it to this list!
