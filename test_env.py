from warehouse_environment import WarehouseFleetEnvironment

# ===== DUMMY CLASSES =====
class DummyAction:
    def __init__(self):
        self.robot_commands = {}

class DummyCommand:
    def __init__(self, command_type, target_package_id=None):
        self.command_type = command_type
        self.target_package_id = target_package_id


# ===== GREEDY AGENT =====
def greedy_action(obs):
    action = DummyAction()

    for robot in obs["robots"]:
        # agar robot idle hai to koi pending package assign karo
        if robot["status"] == "idle":
            for pkg in obs["packages"]:
                if pkg["status"] == "pending":
                    action.robot_commands[robot["id"]] = DummyCommand(
                        "ASSIGN_TASK", pkg["id"]
                    )
                    break

    return action


# ===== MAIN TEST =====

env = WarehouseFleetEnvironment()

obs = env.reset()

print("========== INITIAL STATE ==========")
print(obs)

# ===== STEP 1: NO ACTION TEST =====
print("\n========== NO ACTION TEST ==========")
for i in range(5):
    action = DummyAction()
    obs = env.step(action)
    print(f"Step {i}: Reward={obs['reward']}")


# ===== STEP 2: MANUAL ACTION TEST =====
print("\n========== MANUAL ACTION TEST ==========")

action = DummyAction()
action.robot_commands = {
    "robot_0": DummyCommand("ASSIGN_TASK", "pkg_0")
}

for i in range(50):
    obs = env.step(action)

    print(f"\nStep {i}")
    
    for robot in obs["robots"]:
        print(f"{robot['id']} -> Pos:{robot['position']} Status:{robot['status']}")

    for pkg in obs["packages"]:
        print(f"{pkg['id']} -> Status:{pkg['status']}")

    print(f"Reward: {obs['reward']}")

    # stop if done
    if obs["done"]:
        print("Episode Finished")
        break


# ===== STEP 3: GREEDY AGENT TEST =====
print("\n========== GREEDY AGENT TEST ==========")

obs = env.reset()

for i in range(100):
    action = greedy_action(obs)
    obs = env.step(action)

    print(f"\nStep {i}")
    
    for robot in obs["robots"]:
        print(f"{robot['id']} -> Pos:{robot['position']} Status:{robot['status']}")

    for pkg in obs["packages"]:
        print(f"{pkg['id']} -> Status:{pkg['status']}")

    print(f"Reward: {obs['reward']}")

    if obs["done"]:
        print("\n🎉 ALL PACKAGES DELIVERED or MAX STEPS REACHED")
        break