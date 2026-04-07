# RL-Based Multi-Robot Fleet Management System
## Overview
This project implements a Reinforcement Learning-based Fleet Management System in a 2D grid
environment. Multiple robots navigate, pick up boxes, deliver them to drop zones, avoid collisions,
and manage battery efficiently.
---
## Environment Design
### Grid Representation
- 0: Free Space
- 1: Obstacle
- 2: Charging Station
- 3: Box (Pickup)
- 4: Drop Location
- 5: Robot
---
## Robot Capabilities
- Movement in 4 directions + stay
- Battery decreases every step
- Picks up and drops boxes
- Uses LiDAR-like local perception
- Avoids collisions with obstacles and other robots
---
## Core Features
### 1. Multi-Robot Coordination
- Multiple robots operate simultaneously
- Collision avoidance using:
- swap detection
- destination conflict handling
### 2. Task Allocation
- Robots automatically assigned nearest target:
- Box if not carrying
- Drop if carrying
### 3. Battery Management
- Battery decreases per step
- If battery ≤ 40%, robot prioritizes charging
- Charging stations restore battery
---
## Reward System
### Positive Rewards
- +50 → Successful delivery
- +5 → Picking up box
- +200 → All deliveries completed
- +1 → Charging when battery is low
- +distance improvement reward → moving closer to target
### Negative Rewards
- -0.5 → Collision / invalid move
- -50 → Battery depletion (episode ends)
---
## Observation Space
Each robot receives:
- 5x5 local grid (LiDAR-like view)
- Battery level
- Carrying status
- Relative position to:
- target
- nearest charger
Total: 31-dimensional vector
---
## Action Space
- 0: Up
- 1: Down
- 2: Left
- 3: Right
- 4: Stay
---
## Difficulty Levels
### Easy (easy_delivery)
- 1 box
- Low battery drain (0.5)
- Simple navigation
### Medium (multi_order)
- 4 boxes
- Moderate battery drain (1.5)
- Requires coordination
### Hard (hard_fleet_management)
- 8 boxes
- High battery drain (3.0)
- Complex coordination + charging optimization
---
## Episode Termination
- All boxes delivered
- Battery reaches 0
- Max steps reached (100)
---
## Key Challenges Solved
- Multi-agent coordination
- Dynamic task allocation
- Energy-aware planning
- Collision avoidance
---
## Conclusion
This system demonstrates how reinforcement learning can be used to build intelligent, autonomous
fleet management systems similar to real-world warehouse robots.