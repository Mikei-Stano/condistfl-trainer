from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    CondistFL Split Data Input Model
    """
    kidney_data_path: str = Field(
        description="Path to kidney (KiTS19) dataset directory from DataLoader",
        json_schema_extra={"from_upstream": "always"}
    )
    liver_data_path: str = Field(
        description="Path to liver dataset directory from DataLoader",
        json_schema_extra={"from_upstream": "always"}
    )
    pancreas_data_path: str = Field(
        description="Path to pancreas dataset directory from DataLoader",
        json_schema_extra={"from_upstream": "always"}
    )
    spleen_data_path: str = Field(
        description="Path to spleen dataset directory from DataLoader",
        json_schema_extra={"from_upstream": "always"}
    )
    num_folds: int = Field(
        default=3,
        description="Number of cross-validation folds",
        json_schema_extra={"from_upstream": "never"}
    )
    fold_index: int = Field(
        default=0,
        description="Which fold to use as validation (0-based index)",
        json_schema_extra={"from_upstream": "never"}
    )


class OutputModel(BaseModel):
    """
    CondistFL Split Data Output Model
    """
    kidney_data_root: str = Field(
        description="Path to fold-specific kidney dataset with datalist.json"
    )
    liver_data_root: str = Field(
        description="Path to fold-specific liver dataset with datalist.json"
    )
    pancreas_data_root: str = Field(
        description="Path to fold-specific pancreas dataset with datalist.json"
    )
    spleen_data_root: str = Field(
        description="Path to fold-specific spleen dataset with datalist.json"
    )
    fold_index: int = Field(
        description="The fold index used for validation"
    )
    num_folds: int = Field(
        description="Total number of folds"
    )
    message: str = Field(
        description="Status message"
    )
