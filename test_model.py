from stable_baselines3 import PPO
from gym_env import WarehouseGymEnv

env = WarehouseGymEnv()

model = PPO.load("warehouse_model")

obs, _ = env.reset()

for i in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    print(f"\nStep {i}, Reward: {reward}")

    if done:
        print("🎉 Episode Finished")
        break