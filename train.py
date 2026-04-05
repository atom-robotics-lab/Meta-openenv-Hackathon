from stable_baselines3 import PPO
from gym_env import WarehouseGymEnv

env = WarehouseGymEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
)

model.learn(total_timesteps=1000000)

model.save("warehouse_model")