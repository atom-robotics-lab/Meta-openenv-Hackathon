import argparse
import uvicorn

from openenv.core.env_server.http_server import create_app
# ✅ Use absolute imports (clean + correct for `-m` execution)
from models import FmsAction, FmsObservation
from server.fms_env_environment import FmsEnvironment


# Create OpenEnv app
app = create_app(
    FmsEnvironment,
    FmsAction,
    FmsObservation,
    env_name="fms_warehouse_fleet",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the OpenEnv server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    main(port=args.port)