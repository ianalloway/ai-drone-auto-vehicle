# AGENTS.md - AI Drone & Autonomous Vehicle

## Overview
AI-powered autonomous vehicle and drone control platform. Computer vision, path planning, and real-time control systems.

## Tech Stack
- **Language:** Python 3
- **Dependencies:** See requirements.txt
- **Containerization:** Docker + docker-compose

## Commands
```bash
pip install -r requirements.txt    # Install dependencies
docker-compose up                  # Run via Docker
```

## Project Structure
```
src/                 # Core source code
config/              # Configuration files
examples/            # Example scripts and demos
docs/                # Documentation
Dockerfile           # Container definition
docker-compose.yml   # Multi-container setup
SECURITY.md          # Security policy
```

## Key Conventions
- Has GitHub Actions CI (.github/)
- Security-sensitive project (see SECURITY.md)

## Owner
Ian Alloway (@ianalloway)
