import asyncio
import os
from typing import List
from openai import OpenAI

from models import FmsAction, FmsObservation
from server.fms_env_environment import FmsEnvironment as FmsEnv

# ================= CONFIG =================
API_KEY = os.environ["API_KEY"]          # ✅ MUST (validator injects)
API_BASE_URL = os.environ["API_BASE_URL"]

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.5
MAX_TOTAL_REWARD = 10.0

# ✅ MUST match openenv.yaml
TASKS = ["easy_delivery", "multi_order", "hard_fleet"]

# ================= LOGGING =================
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()}", flush=True)

def log_end(success, steps, score):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f}", flush=True)

# ================= ACTION PARSER =================
def parse_actions(text: str, num_robots: int) -> List[int]:
    try:
        nums = list(map(int, text.strip().split()))
        if len(nums) < num_robots:
            nums += [4] * (num_robots - len(nums))
        return nums[:num_robots]
    except:
        return [4] * num_robots

# ================= MODEL CALL =================
def get_action(client, obs: FmsObservation) -> FmsAction:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Return only integers like: 0 3 (one per robot). No text."
                },
                {
                    "role": "user",
                    "content": f"Observations: {obs.observations}"
                }
            ],
            temperature=0.2,
            max_tokens=50,
        )

        text = (response.choices[0].message.content or "").strip()

        actions = parse_actions(text, len(obs.observations))

        return FmsAction(actions=actions)

    except Exception as e:
        print("[MODEL ERROR]", e, flush=True)
        return FmsAction(actions=[4] * len(obs.observations))  # fallback

# ================= MAIN =================
async def main():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    # 🔥 RUN ALL TASKS
    for task in TASKS:

        env = FmsEnv(task_id=task)
        obs = env.reset()

        rewards = []
        success = False
        score = 0.0

        # ✅ START LOG (per task)
        log_start(task, "fms_warehouse_fleet", MODEL_NAME)

        try:
            for step in range(1, MAX_STEPS + 1):

                action_obj = get_action(client, obs)
                actions = action_obj.actions

                obs = env.step(action_obj)

                reward = obs.reward or 0.0
                done = obs.done

                rewards.append(reward)

                # ✅ STEP LOG
                log_step(step, actions, reward, done)

                if done:
                    break

            # ✅ SCORE FIX (0 < score < 1)
            total = sum(rewards)
            raw_score = total / MAX_TOTAL_REWARD
            score = max(0.01, min(raw_score, 0.99))

            success = score >= SUCCESS_THRESHOLD

        finally:
            env.close()

            # ✅ END LOG (per task)
            log_end(success, len(rewards), score)

# ================= RUN =================
if __name__ == "__main__":
    asyncio.run(main())