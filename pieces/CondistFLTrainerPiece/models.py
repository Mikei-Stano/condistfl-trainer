from pydantic import BaseModel, Field
from typing import Optional, Dict


class InputModel(BaseModel):
    """
    CondistFL Training Input Model
    """
    num_rounds: int = Field(
        default=3,
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
        default="/app/data/KiTS19",
        description="Path to kidney dataset root",
        json_schema_extra={"from_upstream": "never"}
    )
    data_root_liver: str = Field(
        default="/app/data/Liver",
        description="Path to liver dataset root",
        json_schema_extra={"from_upstream": "never"}
    )
    data_root_pancreas: str = Field(
        default="/app/data/Pancreas",
        description="Path to pancreas dataset root",
        json_schema_extra={"from_upstream": "never"}
    )
    data_root_spleen: str = Field(
        default="/app/data/Spleen",
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
    best_global_model_path: str = Field(
        description="Path to the best global model checkpoint"
    )
    global_model_path: str = Field(
        description="Path to the final global model checkpoint"
    )
    best_local_models: Dict[str, str] = Field(
        description="Paths to best local models for each client",
        default_factory=dict
    )
    cross_site_validation_results: str = Field(
        description="Path to cross-site validation results YAML file"
    )
    training_complete: bool = Field(
        description="Whether training completed successfully"
    )
    num_rounds_completed: int = Field(
        description="Number of rounds completed"
    )
    validation_metrics: Dict[str, float] = Field(
        description="Summary of validation metrics (average Dice scores)",
        default_factory=dict
    )
    message: str = Field(
        description="Status message"
    )
