import os
import json
import shutil
import base64
from pathlib import Path
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


class CondistFLDataLoaderPiece(BasePiece):
    """
    Loads baked-in CondistFL multi-organ segmentation data and copies it
    to shared storage so downstream pieces can access it.

    Expects four dataset folders (KiTS19, Liver, Pancreas, Spleen) each
    containing a datalist.json and .nii.gz files.
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
        results_dir = Path(getattr(self, "results_path", "/tmp"))
        results_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Loading data from {data_dir}")

        output_paths = {}
        summary_lines = []

        for folder, label in self.DATASETS.items():
            src = data_dir / folder
            dst = results_dir / folder

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

            # Copy entire folder to shared storage
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src), str(dst))

            output_paths[label] = str(dst)
            summary_lines.append(
                f"{label} ({folder}): {n_samples} samples, {len(nifti_files)} NIfTI files"
            )
            self.logger.info(f"Copied {folder} -> {dst}")

        # Display result in Domino UI
        summary_text = "CondistFL Data Loaded\n" + "\n".join(summary_lines)
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
            message=f"Loaded 4 datasets to {results_dir}",
        )
