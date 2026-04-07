import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI

# ✅ Import your specific FMS models and environment
from fms_env.models import FmsAction, FmsObservation
from fms_env.server.fms_env_environment import FmsEnvironment as FmsEnv

# Environment Setup
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Benchmark Config
TASK_NAME = os.getenv("FMS_TASK", "warehouse-delivery")
BENCHMARK = os.getenv("FMS_BENCHMARK", "fms_warehouse_fleet")
MAX_STEPS = 20  # Increased for multi-robot navigation
TEMPERATURE = 0.2 # Lower temperature for more logical planning
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.5  # Need at least half the boxes delivered

# Normalized Score Logic: Adjust based on your max possible reward
MAX_TOTAL_REWARD = 10.0 

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Fleet Management AI controlling multiple robots in a 2D warehouse grid.
    GOAL: Navigate robots to pick up boxes and deliver them to drop zones.
    CONSTRAINTS: Avoid collisions and monitor battery levels.
    
    Current actions available: "move_north", "move_south", "move_east", "move_west", "pick_up", "drop_off", "wait".
    
    Reply with exactly one action in this format: robot_id:action_type
    Example: 1:move_north
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs: FmsObservation, history: List[str]) -> str:
    # Summarize the warehouse state for the LLM
    robot_info = "\n".join([f"Robot {rid}: Pos {pos}, Battery {obs.battery_levels.get(rid)}%" 
                            for rid, pos in obs.robot_positions.items()])
    
    return textwrap.dedent(
        f"""
        Step: {step}
        --- Current Warehouse State ---
        {robot_info}
        Box Locations: {obs.box_locations}
        
        Recent History:
        {" | ".join(history[-3:]) if history else "None"}
        
        Decide the next action for one robot:
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, obs: FmsObservation, history: List[str]) -> FmsAction:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        
        # Parse "robot_id:action" string (e.g., "1:move_north")
        parts = raw_text.split(":")
        rid = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        act = parts[1] if len(parts) > 1 else "wait"
        
        return FmsAction(robot_id=rid, action_type=act)
    except Exception as exc:
        print(f"[DEBUG] Model failed: {exc}", flush=True)
        return FmsAction(robot_id=0, action_type="wait")

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize your environment
    # Note: If not using Docker, use FmsEnv() directly. 
    # If using Docker, keep the from_docker_image method.
    env = await FmsEnv.from_docker_image(IMAGE_NAME) if IMAGE_NAME else FmsEnv()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset returns the first observation
        obs = await env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            action_obj = get_model_action(client, step, obs, history)
            
            # Execute step
            result = await env.step(action_obj)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            
            rewards.append(reward)
            steps_taken = step
            
            action_str = f"{action_obj.robot_id}:{action_obj.action_type}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            history.append(f"S{step}: {action_str} (R:{reward})")

            if done:
                break

        # Calculate final score (normalized 0.0 to 1.0)
        total_r = sum(rewards)
        score = min(max(total_r / MAX_TOTAL_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())