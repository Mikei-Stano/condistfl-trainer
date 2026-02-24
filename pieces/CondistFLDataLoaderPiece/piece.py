import os
import json
import base64
from pathlib import Path
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


class CondistFLDataLoaderPiece(BasePiece):
    """
    Verifies baked-in CondistFL multi-organ segmentation data and outputs
    the dataset paths for downstream pieces.

    Expects four dataset folders (KiTS19, Liver, Pancreas, Spleen) each
    containing a datalist.json and .nii.gz files at /data inside the container.
    """

    # Dataset folder name -> friendly label
    DATASETS = {
        "KiTS19": "kidney",
        "Liver": "liver",
        "Pancreas": "pancreas",
        "Spleen": "spleen",
    }

    def piece_function(self, input_data: InputModel) -> OutputModel:
        data_dir = Path(input_data.data_dir)

        self.logger.info(f"Verifying data at {data_dir}")

        output_paths = {}
        summary_lines = []

        for folder, label in self.DATASETS.items():
            src = data_dir / folder

            if not src.exists():
                raise FileNotFoundError(f"Dataset folder not found: {src}")

            datalist_file = src / "datalist.json"
            if not datalist_file.exists():
                raise FileNotFoundError(f"datalist.json not found in {src}")

            # Read datalist to report counts
            with open(datalist_file, "r") as f:
                datalist = json.load(f)
            n_samples = len(datalist.get("training", []))

            # Count .nii.gz files
            nifti_files = list(src.glob("*.nii.gz"))

            output_paths[label] = str(src)
            summary_lines.append(
                f"{label} ({folder}): {n_samples} samples, {len(nifti_files)} NIfTI files"
            )
            self.logger.info(f"Verified {folder}: {n_samples} samples, {len(nifti_files)} NIfTI files")

        # Display result in Domino UI
        summary_text = "CondistFL Data Verified\n" + "\n".join(summary_lines)
        self.display_result = {
            "file_type": "txt",
            "base64_content": base64.b64encode(
                summary_text.encode("utf-8")
            ).decode("utf-8"),
        }

        return OutputModel(
            kidney_data_path=output_paths["kidney"],
            liver_data_path=output_paths["liver"],
            pancreas_data_path=output_paths["pancreas"],
            spleen_data_path=output_paths["spleen"],
            message=f"Verified 4 datasets at {data_dir}",
        )
