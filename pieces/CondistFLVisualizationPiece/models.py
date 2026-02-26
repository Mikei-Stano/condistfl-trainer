from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class InputModel(BaseModel):
    """
    CondistFL Visualization Input Model
    """
    training_complete: bool = Field(
        description="Whether training completed successfully",
        json_schema_extra={"from_upstream": "always"}
    )
    num_rounds_completed: int = Field(
        description="Number of federated learning rounds completed",
        json_schema_extra={"from_upstream": "always"}
    )
    validation_metrics: Dict[str, float] = Field(
        description="Summary of validation metrics (Dice scores per client)",
        default_factory=dict,
        json_schema_extra={"from_upstream": "always"}
    )
    client_metrics: Dict[str, Dict[str, List[Dict[str, float]]]] = Field(
        description=(
            "Per-client TensorBoard scalars. "
            "Structure: {client: {tag: [{step, value}, ...]}}"
        ),
        default_factory=dict,
        json_schema_extra={"from_upstream": "always"}
    )
    server_metrics: Dict[str, List[Dict[str, float]]] = Field(
        description="Server TensorBoard scalars",
        default_factory=dict,
        json_schema_extra={"from_upstream": "always"}
    )
    cross_val_data: Optional[List[Dict[str, Any]]] = Field(
        description="Parsed cross-site validation results",
        default=None,
        json_schema_extra={"from_upstream": "always"}
    )


class OutputModel(BaseModel):
    """
    CondistFL Visualization Output Model
    """
    charts_dir: str = Field(
        description="Directory containing saved chart PNG files"
    )
    summary: str = Field(
        description="Text summary of training results"
    )
    message: str = Field(
        description="Status message"
    )
