import gymnasium as gym
import numpy as np
from warehouse_environment import WarehouseFleetEnvironment

class WarehouseGymEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.env = WarehouseFleetEnvironment()

        # Observation: simple flatten
        self.observation_space = gym.spaces.Box(
    low=-1, high=1, shape=(14,), dtype=np.float32
)

        # Action: 2 robots, 6 possible actions each
        self.action_space = gym.spaces.MultiDiscrete([6, 6])

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return self._obs_to_array(obs), {}

    def step(self, action):
        act = self._convert_action(action)
        obs = self.env.step(act)

        return (
            self._obs_to_array(obs),
            obs["reward"],
            obs["done"],
            False,
            {}
        )

    def _obs_to_array(self, obs):
        arr = []

        for r in obs["robots"]:
            arr.extend([r["position"][0], r["position"][1]])

        for p in obs["packages"]:
            arr.extend([p["pickup"][0], p["pickup"][1]])

        return np.array(arr[:50], dtype=np.float32)

    def _convert_action(self, action):
        class DummyAction:
            def __init__(self):
                self.robot_commands = {}

        class DummyCommand:
            def __init__(self, command_type, target_package_id=None):
                self.command_type = command_type
                self.target_package_id = target_package_id

        act = DummyAction()

        for i, a in enumerate(action):
            if a > 0:
                act.robot_commands[f"robot_{i}"] = DummyCommand(
                    "ASSIGN_TASK", f"pkg_{a-1}"
                )

        return act