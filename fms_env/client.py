# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Dict, Any
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import FmsAction, FmsObservation
except (ImportError, ModuleNotFoundError):
    from models import FmsAction, FmsObservation

class FmsEnv(EnvClient[FmsAction, FmsObservation, State]):
    """
    Client for the Warehouse Fleet Management System.
    Connects an AI Agent to the FmsEnvironment server.
    """

    def _step_payload(self, action: FmsAction) -> Dict[str, Any]:
        """Converts the FmsAction Pydantic model to a JSON-safe dict."""
        return {
            "actions": action.actions # List of ints: [0, 1, 4...]
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FmsObservation]:
        """Parses the server's JSON response back into our Pydantic Observation."""
        obs_data = payload.get("observation", {})
        
        # Reconstruct the Observation model
        observation = FmsObservation(
            observations=obs_data.get("observations", []),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            message=obs_data.get("message", ""),
            grid=obs_data.get("grid", None)
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Standard OpenEnv state parsing."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )