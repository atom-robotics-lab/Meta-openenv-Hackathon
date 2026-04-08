# Multi-Robot Fleet Management System (FMS)

## Overview
This project implements a **Fleet Management System (FMS)** in a 2D grid-based warehouse environment.  
Multiple robots navigate, pick up boxes, deliver them to drop zones, avoid collisions, and manage battery levels.



---

## What is FMS (Fleet Management System)?

A Fleet Management System is responsible for:
- Assigning tasks to robots
- Planning paths for navigation
- Avoiding collisions
- Managing resources like battery

In real-world warehouses (like Amazon), FMS ensures efficient and safe robot coordination.

---

## Without RL (Current Implementation)

In this project, robots operate using:

### 🔹 Path Planning
- Grid-based movement (Up, Down, Left, Right)
- Local navigation using 5x5 observation window
- Distance-based movement towards target

### 🔹 Task Allocation
- If robot is **not carrying** → nearest **box assigned**
- If robot is **carrying** → nearest **drop zone assigned**

### 🔹 Collision Avoidance
- Swap detection (robots swapping positions blocked)
- Destination conflict handling (multiple robots same cell blocked)

### 🔹 Battery Handling (Basic)
- Battery decreases every step
- Charging stations restore battery
- No intelligent scheduling (rule-based)

👉 This is mostly **rule-based + heuristic system**

---

## With RL (Future Scope)

If Reinforcement Learning is applied, system becomes much smarter:

### 🚀 Advanced Capabilities with RL
- Learn **optimal task allocation**
- Minimize **total travel time**
- Reduce **collisions automatically**
- Learn **when to charge vs when to deliver**
- Handle **multi-robot coordination dynamically**

### 🔋 Battery Optimization
- Decide best time to go to charger
- Avoid unnecessary charging
- Balance delivery vs charging

### 📉 Performance Improvements
- Fewer collisions
- Faster deliveries
- Better resource utilization

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
- Move in 4 directions + stay
- Pick and drop boxes
- Battery consumption per step
- Local perception (5x5 grid like LiDAR)
- Collision avoidance

---

## Core Features

### 1. Multi-Robot Coordination
- Multiple robots operate simultaneously
- Collision handling:
  - Swap conflict prevention
  - Destination conflict resolution

### 2. Task Allocation (Rule-Based)
- Nearest box assigned if not carrying
- Nearest drop assigned if carrying

### 3. Battery Management
- Battery decreases each step
- Charging station restores battery
- Low battery threshold triggers charging (basic logic)

---

## Reward System (for RL compatibility)

### Positive Rewards
- +50 → Delivery completed  
- +5 → Box picked  
- +200 → All deliveries completed  
- +1 → Charging when battery is low  
- +distance improvement → moving closer to target  

### Negative Rewards
- -0.5 → Collision / invalid move  
- -50 → Battery depletion  

---

## Observation Space
Each robot gets a **31-dimensional vector**:
- 5x5 local grid (25 values)
- Battery level
- Carrying status
- Relative target position
- Relative charger position

---

## Action Space
- 0: Up  
- 1: Down  
- 2: Left  
- 3: Right  
- 4: Stay  

---

## Difficulty Levels

### Easy (`easy_delivery`)
- 1 box  
- Low battery drain  
- Simple navigation  

### Medium (`multi_order`)
- 4 boxes  
- Moderate battery drain  
- Requires coordination  

### Hard (`hard_fleet`)
- 8 boxes  
- High battery drain  
- Complex coordination + charging  

---

## Episode Termination
- All boxes delivered  
- Battery reaches 0  
- Max steps reached  

---

## Key Contributions
- Designed a **custom multi-robot simulation environment**
- Implemented **collision-safe movement logic**
- Built **task allocation and navigation system**
- Created **RL-compatible observation + reward system**

---

## Conclusion

This project demonstrates how a **Fleet Management System can be built using rule-based planning**.  

While current implementation uses **heuristics (path planning + allocation)**,  
it can be extended with **Reinforcement Learning** to achieve:

- Smarter decision making  
- Better coordination  
- Efficient energy usage  
- Reduced collisions  

👉 This makes it a strong foundation for real-world warehouse robotics systems.