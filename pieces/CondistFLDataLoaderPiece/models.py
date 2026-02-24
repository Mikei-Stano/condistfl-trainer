from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    CondistFL Data Loader Input Model
    """
    data_dir: str = Field(
        default="/data",
        description="Root directory containing the baked-in dataset folders (KiTS19, Liver, Pancreas, Spleen)",
        json_schema_extra={"from_upstream": "never"}
    )


class OutputModel(BaseModel):
    """
    CondistFL Data Loader Output Model
    """
    kidney_data_path: str = Field(
        description="Path to kidney (KiTS19) dataset directory"
    )
    liver_data_path: str = Field(
        description="Path to liver dataset directory"
    )
    pancreas_data_path: str = Field(
        description="Path to pancreas dataset directory"
    )
    spleen_data_path: str = Field(
        description="Path to spleen dataset directory"
    )
    message: str = Field(
        description="Status message"
    )
