# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import List, Optional, Dict
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class FmsAction(Action):
    """
    The Agent (LLM) will see this description. 
    It tells the AI exactly what numbers to send.
    """
    actions: List[int] = Field(
        ..., 
        description="List of integers for each robot: 0:Up, 1:Down, 2:Left, 3:Right, 4:Wait. Length must match number of robots."
    )

class FmsObservation(Observation):
    """
    This is what the Agent receives after every step.
    """
    observations: List[List[float]] = Field(
        ...,
        description="List of flattened 31-element vectors (5x5 local grid + battery + carrying + target_dist + charger_dist) for each robot."
    )
    reward: float = Field(default=0.0, description="Cumulative reward for the fleet in this step.")
    done: bool = Field(default=False, description="True if all boxes are delivered or battery is exhausted.")
    message: str = Field(default="", description="Status updates (e.g., 'Collision detected', 'Box picked up').")
    
    grid: Optional[List[List[int]]] = Field(
        None, 
        description="10x10 integer representation of the warehouse floor."
    )