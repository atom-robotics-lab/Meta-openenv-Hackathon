import numpy as np
import random
from uuid import uuid4
from collections import deque
from gymnasium import spaces
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FmsAction, FmsObservation
except ImportError:
    from models import FmsAction, FmsObservation

class FmsEnvironment(Environment):
    """
    Meta OpenEnv Compliant Fleet Management System (FMS).
    Simulates warehouse robots fulfilling orders while managing battery and collisions.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    FREE, OBSTACLE, CHARGER, BOX, DROP, ROBOT = 0, 1, 2, 3, 4, 5
    
    ROBOT_INIT_POSITIONS = [(0, 0), (0, 9)]
    CHARGER_POSITIONS = [(9, 4), (9, 5)]

    def __init__(self, task_id: str = "easy_delivery"):
        super().__init__()
        self.task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.rows, self.cols = 10, 10
        self.num_robots = 2
        self.recharge_rate = 10.0
        self.low_battery_threshold = 40.0

        if task_id == "easy_delivery":
            self.num_boxes = 1
            self.battery_drain = 0.5
        elif task_id == "multi_order":
            self.num_boxes = 4
            self.battery_drain = 1.5
        else: # "hard_fleet_management"
            self.num_boxes = 8
            self.battery_drain = 3.0

        self.grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.robots = []
        self.delivered_count = 0
        self.collision_count = 0
        self.pickup_count = 0

    def reset(self, seed=None, options=None) -> FmsObservation:
        self._state.step_count = 0
        self._state.episode_id = str(uuid4())
        self.grid.fill(self.FREE)
        self.delivered_count = 0
        self.pickup_count = 0
        self.collision_count = 0

        for pos in self.CHARGER_POSITIONS:
            self.grid[pos] = self.CHARGER

        self.robots = []
        for i in range(self.num_robots):
            pos = self.ROBOT_INIT_POSITIONS[i % len(self.ROBOT_INIT_POSITIONS)]
            while self.grid[pos] != self.FREE:
                pos = (pos[0], pos[1] + (1 if pos[1] < 9 else -1))
            
            self.robots.append({
                'id': i,
                'pos': pos,
                'battery': 100.0,
                'carrying': False,
                'task_target': None,
            })
            self.grid[pos] = self.ROBOT

        self._place_objects(count=int(self.rows * self.cols * 0.10), val=self.OBSTACLE)
        self._place_objects(count=self.num_boxes, val=self.BOX)
        self._place_objects(count=1, val=self.DROP)

        self._assign_task_targets()

        return self._get_fms_obs("Environment Reset Successful")

    def step(self, action: FmsAction) -> FmsObservation:
        self._state.step_count += 1
        total_reward = 0.0
        terminated = False

        robot_actions = action.actions # list[int]

        intended = []
        for i, act in enumerate(robot_actions):
            if i < len(self.robots):
                intended.append(self._calculate_move(self.robots[i]['pos'], act))
            else:
                intended.append(None)

        swap_blocked = set()
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                if intended[i] == self.robots[j]['pos'] and intended[j] == self.robots[i]['pos']:
                    swap_blocked.add(i); swap_blocked.add(j)

        dest_count = {}
        for pos in intended:
            if pos: dest_count[pos] = dest_count.get(pos, 0) + 1
        dest_blocked = {pos for pos, cnt in dest_count.items() if cnt > 1}

        for i, robot in enumerate(self.robots):
            act = robot_actions[i] if i < len(robot_actions) else 4
            new_pos = intended[i]
            old_dist = self._dist_to_target(robot)

            if (new_pos and i not in swap_blocked and new_pos not in dest_blocked 
                and self._is_valid_move(new_pos, i)):
                self.grid[robot['pos']] = self.FREE
                robot['pos'] = new_pos
                self.grid[new_pos] = self.ROBOT
            else:
                if act != 4: # Penalize unintended collisions
                    total_reward -= 0.5
                    self.collision_count += 1

            cell = self._underlying_cell(robot['pos'])
            if cell == self.CHARGER:
                robot['battery'] = min(100.0, robot['battery'] + self.recharge_rate)
                if robot['battery'] < self.low_battery_threshold:
                    total_reward += 1.0 # Positive reinforcement for charging when low
            else:
                robot['battery'] -= self.battery_drain

            if cell == self.BOX and not robot['carrying']:
                robot['carrying'] = True
                self.pickup_count += 1
                total_reward += 5.0
                robot['task_target'] = self._find_nearest(robot['pos'], self.DROP)
            elif cell == self.DROP and robot['carrying']:
                robot['carrying'] = False
                self.delivered_count += 1
                total_reward += 50.0
                robot['task_target'] = self._find_nearest(robot['pos'], self.BOX)

            new_dist = self._dist_to_target(robot)
            if old_dist is not None and new_dist is not None:
                total_reward += (old_dist - new_dist) * 0.1

            if robot['battery'] <= 0:
                terminated = True
                total_reward -= 50.0

        no_boxes_left = not np.any(self.grid == self.BOX)
        all_delivered = all(not r['carrying'] for r in self.robots)
        
        if (no_boxes_left and all_delivered) or self._state.step_count >= 100:
            terminated = True
            if no_boxes_left and all_delivered:
                total_reward += 200.0

        return self._get_fms_obs("Step Successful", reward=total_reward, done=terminated)

    def _get_obs_list(self) -> list:
        """Generates the 31-element vector for each robot."""
        obs_list = []
        for robot in self.robots:
            r, c = robot['pos']
            padded = np.pad(self.grid, 2, mode='constant', constant_values=self.OBSTACLE)
            crop = padded[r:r+5, c:c+5].flatten().astype(np.float32) / 5.0
            
            target = robot['task_target'] or (0, 0)
            charger = self._find_nearest(robot['pos'], self.CHARGER) or (0, 0)
            
            stats = [
                robot['battery'] / 100.0,
                1.0 if robot['carrying'] else 0.0,
                (target[0] - r) / self.rows,
                (target[1] - c) / self.cols,
                (charger[0] - r) / self.rows,
                (charger[1] - c) / self.cols
            ]
            obs_list.append(np.concatenate([crop, stats]).tolist())
        return obs_list

    def _get_fms_obs(self, msg: str, reward: float = 0.0, done: bool = False) -> FmsObservation:
        return FmsObservation(
            observations=self._get_obs_list(),
            reward=float(reward),
            done=bool(done),
            message=msg,
            grid=self.grid.tolist()
        )

    def _calculate_move(self, pos, action):
        r, c = pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        dr, dc = moves.get(action, (0, 0))
        return (max(0, min(self.rows-1, r + dr)), max(0, min(self.cols-1, c + dc)))

    def _is_valid_move(self, pos, robot_id):
        r, c = pos
        cell = self.grid[r, c]
        if cell == self.OBSTACLE: return False
        if cell == self.ROBOT:
            return any(rob['pos'] == pos and rob['id'] == robot_id for rob in self.robots)
        return True

    def _underlying_cell(self, pos):
        if pos in self.CHARGER_POSITIONS: return self.CHARGER
        return self.grid[pos]

    def _find_nearest(self, pos, cell_type):
        q = deque([pos])
        visited = {pos}
        while q:
            r, c = q.popleft()
            if self.grid[r, c] == cell_type or (cell_type == self.CHARGER and (r, c) in self.CHARGER_POSITIONS):
                if (r, c) != pos: return (r, c)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                    visited.add((nr, nc)); q.append((nr, nc))
        return None

    def _dist_to_target(self, robot):
        if not robot['task_target']: return None
        return abs(robot['task_target'][0] - robot['pos'][0]) + abs(robot['task_target'][1] - robot['pos'][1])

    def _assign_task_targets(self):
        for r in self.robots:
            r['task_target'] = self._find_nearest(r['pos'], self.DROP if r['carrying'] else self.BOX)

    def _place_objects(self, count, val):
        placed = 0
        while placed < count:
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if self.grid[r, c] == self.FREE and (r, c) not in self.CHARGER_POSITIONS:
                self.grid[r, c] = val
                placed += 1

    @property
    def state(self) -> State:
        return self._state