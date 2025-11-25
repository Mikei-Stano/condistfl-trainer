from pydantic import BaseModel, Field
from typing import Optional, List


class InputModel(BaseModel):
    """
    CondistFL Training Input Model
    """
    num_rounds: int = Field(
        default=120,
        description="Number of federated learning rounds",
        json_schema_extra={"from_upstream": "never"}
    )
    steps_per_round: int = Field(
        default=1000,
        description="Training steps per round",
        json_schema_extra={"from_upstream": "never"}
    )
    clients: str = Field(
        default="liver,spleen,pancreas,kidney",
        description="Comma-separated list of client names",
        json_schema_extra={"from_upstream": "never"}
    )
    gpus: str = Field(
        default="0,1,2,3",
        description="Comma-separated GPU IDs to use",
        json_schema_extra={"from_upstream": "never"}
    )
    data_root_kidney: str = Field(
        default="/app/data_sampled/KiTS19",
        description="Path to kidney dataset root",
        json_schema_extra={"from_upstream": "never"}
    )
    data_root_liver: str = Field(
        default="/app/data_sampled/Liver",
        description="Path to liver dataset root",
        json_schema_extra={"from_upstream": "never"}
    )
    data_root_pancreas: str = Field(
        default="/app/data_sampled/Pancreas",
        description="Path to pancreas dataset root",
        json_schema_extra={"from_upstream": "never"}
    )
    data_root_spleen: str = Field(
        default="/app/data_sampled/Spleen",
        description="Path to spleen dataset root",
        json_schema_extra={"from_upstream": "never"}
    )
    workspace_dir: str = Field(
        default="/app/workspace",
        description="Directory to save training workspace",
        json_schema_extra={"from_upstream": "never"}
    )


class OutputModel(BaseModel):
    """
    CondistFL Training Output Model
    """
    workspace_dir: str = Field(
        description="Directory containing training results and models"
    )
    best_global_model: str = Field(
        description="Path to the best global model checkpoint"
    )
    training_complete: bool = Field(
        description="Whether training completed successfully"
    )
    num_rounds_completed: int = Field(
        description="Number of rounds completed"
    )
    message: str = Field(
        description="Status message"
    )
