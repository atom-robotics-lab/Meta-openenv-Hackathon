import random
from typing import Dict, Tuple
from dataclasses import dataclass


# ==== DATA STRUCTURES ====

@dataclass
class Robot:
    id: str
    position: Tuple[int, int]
    battery: float
    status: str  # idle, moving, charging
    assigned_package: str = None


@dataclass
class Package:
    id: str
    pickup: Tuple[int, int]
    dropoff: Tuple[int, int]
    status: str  # pending, picked, delivered


# ==== MAIN ENVIRONMENT ====

class WarehouseFleetEnvironment:
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.grid_size = config.get("grid_size", (20, 20))
        self.num_robots = config.get("num_robots", 2)
        self.num_packages = config.get("num_packages", 5)

        self.robots: Dict[str, Robot] = {}
        self.packages: Dict[str, Package] = {}

        self.step_count = 0
        self.max_steps = 500

    # ================= RESET =================
    def reset(self):
        self.step_count = 0

        self.robots = {}
        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Robot(
                id=f"robot_{i}",
                position=self._random_pos(),
                battery=1.0,
                status="idle"
            )

        self.packages = {}
        for i in range(self.num_packages):
            self.packages[f"pkg_{i}"] = Package(
                id=f"pkg_{i}",
                pickup=self._random_pos(),
                dropoff=self._random_pos(),
                status="pending"
            )

        return self._get_observation(reward=0.0, done=False)

    # ================= STEP =================
    def step(self, action):
        self.step_count += 1
        reward = 0

        # ---- APPLY ACTIONS ----
        for robot_id, cmd in action.robot_commands.items():
            robot = self.robots.get(robot_id)
            if not robot:
                continue

            if cmd.command_type == "ASSIGN_TASK":
                pkg = self.packages.get(cmd.target_package_id)

                if pkg and pkg.status == "pending":
                    robot.assigned_package = pkg.id
                    robot.status = "moving"

            elif cmd.command_type == "GOTO_CHARGE":
                robot.status = "charging"

        # ---- MOVE ROBOTS ----
        for robot in self.robots.values():
            if robot.status == "moving" and robot.assigned_package:
                pkg = self.packages[robot.assigned_package]

                target = pkg.pickup if pkg.status == "pending" else pkg.dropoff

                # Distance before move
                old_dist = self._distance(robot.position, target)

                # Move
                robot.position = self._move_towards(robot.position, target)

                # Distance after move
                new_dist = self._distance(robot.position, target)

                # ✅ Dense reward
                reward += (old_dist - new_dist) * 0.5

                # Pickup
                if robot.position == pkg.pickup and pkg.status == "pending":
                    pkg.status = "picked"
                    reward += 2

                # Deliver
                elif robot.position == pkg.dropoff and pkg.status == "picked":
                    pkg.status = "delivered"
                    robot.assigned_package = None
                    robot.status = "idle"
                    reward += 20

            # Charging
            if robot.status == "charging":
                robot.battery = min(1.0, robot.battery + 0.05)
                reward += 0.05

            # Battery drain
            robot.battery -= 0.005

            if robot.battery < 0.2:
                reward -= 0.5

        # ---- COLLISION CHECK ----
        positions = {}
        for robot in self.robots.values():
            if robot.position in positions:
                reward -= 2
            positions[robot.position] = True

        # ---- TIME PENALTY ----
        reward -= 0.01

        # ---- DONE ----
        done = self._check_done()

        if done:
            delivered = sum(p.status == "delivered" for p in self.packages.values())
            total = len(self.packages)
            reward += (delivered / total) * 20

        return self._get_observation(reward, done)

    # ================= HELPERS =================

    def _distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def _random_pos(self):
        return (
            random.randint(0, self.grid_size[0] - 1),
            random.randint(0, self.grid_size[1] - 1),
        )

    def _move_towards(self, pos, target):
        x, y = pos
        tx, ty = target

        if x < tx:
            x += 1
        elif x > tx:
            x -= 1
        elif y < ty:
            y += 1
        elif y > ty:
            y -= 1

        return (x, y)

    def _check_done(self):
        return (
            all(p.status == "delivered" for p in self.packages.values())
            or self.step_count >= self.max_steps
        )

    # ================= OBSERVATION =================

    def _get_observation(self, reward, done):
        return {
            "robots": [
                {
                    "id": r.id,
                    "position": r.position,
                    "battery": r.battery,
                    "status": r.status,
                    "assigned_package": r.assigned_package,
                }
                for r in self.robots.values()
            ],
            "packages": [
                {
                    "id": p.id,
                    "pickup": p.pickup,
                    "dropoff": p.dropoff,
                    "status": p.status,
                }
                for p in self.packages.values()
            ],
            "reward": reward,
            "done": done,
            "step_count": self.step_count,
        }