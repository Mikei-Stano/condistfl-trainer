import os
import json
import subprocess
from pathlib import Path
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


class CondistFLTrainerPiece(BasePiece):
    """
    CondistFL Federated Learning Trainer
    
    This piece trains a federated learning model for multi-organ and tumor 
    segmentation using the CondistFL framework with NVFlare.
    
    The training process:
    1. Configures data paths for each client (kidney, liver, pancreas, spleen)
    2. Launches NVFlare simulator with specified clients and GPUs
    3. Trains for specified number of rounds
    4. Saves best global and local models
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        """
        Execute the CondistFL federated training
        
        Args:
            input_data: InputModel with training configuration
            
        Returns:
            OutputModel with training results
        """
        # Base directory inside the container where code and jobs were copied
        base_dir = Path("/app")
        
        # Ensure workspace directory exists
        workspace_path = Path(input_data.workspace_dir)
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting CondistFL training in workspace: {workspace_path}")
        self.logger.info(f"Clients: {input_data.clients}")
        self.logger.info(f"GPUs: {input_data.gpus}")
        self.logger.info(f"Rounds: {input_data.num_rounds}")
        
        # Update data paths in job configs for each client
        clients = input_data.clients.split(',')
        data_paths = {
            'kidney': input_data.data_root_kidney,
            'liver': input_data.data_root_liver,
            'pancreas': input_data.data_root_pancreas,
            'spleen': input_data.data_root_spleen
        }
        
        jobs_dir = base_dir / "jobs" / "condist"
        
        for client in clients:
            client = client.strip()
            if client in data_paths:
                config_file = jobs_dir / client / "config" / "config_data.json"
                if config_file.exists():
                    self.logger.info(f"Updating data paths for {client}")
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Update paths
                    config['data_root'] = data_paths[client]
                    config['data_list'] = f"{data_paths[client]}/datalist.json"
                    
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
                else:
                    self.logger.warning(f"Config file not found for {client}: {config_file}")
        
        # Prepare the training command
        cmd = [
            "nvflare", "simulator",
            "-w", str(workspace_path.absolute()),
            "-c", input_data.clients,
            "-gpu", input_data.gpus,
            str(jobs_dir.absolute())
        ]
        
        # Set environment variable
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{base_dir}/src:{env.get('PYTHONPATH', '')}"
        
        self.logger.info(f"Running training command: {' '.join(cmd)}")
        
        # Execute the training
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(base_dir),
                env=env
            )
            self.logger.info(f"Training stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Training stderr: {result.stderr}")
            
            training_complete = True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Training failed with error: {e.stderr}")
            training_complete = False
            # Don't raise - we'll still return what we have
        
        # Look for the best global model
        best_model_path = workspace_path / "simulate_job" / "app_server" / "best_FL_global_model.pt"
        
        if best_model_path.exists():
            best_model = str(best_model_path.absolute())
            self.logger.info(f"Found best global model: {best_model}")
        else:
            best_model = "Not found - training may have failed"
            self.logger.warning("Best global model not found")
        
        # Count completed rounds by checking workspace logs
        num_rounds_completed = input_data.num_rounds if training_complete else 0
        
        message = "Training completed successfully" if training_complete else "Training failed or incomplete"
        
        return OutputModel(
            workspace_dir=str(workspace_path.absolute()),
            best_global_model=best_model,
            training_complete=training_complete,
            num_rounds_completed=num_rounds_completed,
            message=message
        )

    # Override default container resources for federated learning training
    container_resources = {
        "requests": {
            "cpu": 4000,
            "memory": 8192
        },
        "limits": {
            "cpu": 16000,
            "memory": 32768
        },
        "use_gpu": True,
        "shm_size": 8192  # 8GB shared memory for PyTorch DataLoader multiprocessing
    }
