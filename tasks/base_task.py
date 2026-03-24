import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseTask(ABC):
    """
    Base class for all evaluation tasks.
    """
    
    def __init__(self, model_name: str, task_name: str, data_root: str = "/DATA/wfr/Q-eval/dataset/AGI-Eval/Q-Eval-100K/"):
        """
        Initialize the base task.
        
        Args:
            model_name: Name of the model being evaluated
            task_name: Name of the task
            data_root: Root directory for dataset
        """
        self.model_name = model_name
        self.task_name = task_name
        self.data_root = data_root
        
        # Set up paths
        self.result_dir = f"/DATA/wfr/Q-eval/baseline/result/{model_name}"
        self.log_dir = "/DATA/wfr/Q-eval/baseline/logs"
        
        # Create directories if they don't exist
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logger
        self.log_file = os.path.join(self.log_dir, f"{model_name}_{task_name}.log")
        self.logger = self._setup_logger()
        
        # Task attributes
        self.task_attributes = {
            "model": model_name,
            "task": task_name,
            "timestamp": None,  # Will be set when saving results
            "metrics": {}
        }
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up a logger for this task.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(f"{self.model_name}_{self.task_name}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers if any
        if logger.handlers:
            logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def save_results(self, results: Any, mission: str) -> None:
        """
        Save results to a JSON file.
        
        Args:
            results: Results to save
            mission: Mission name (used in filename)
        """
        import datetime
        
        # Update task attributes
        self.task_attributes["timestamp"] = datetime.datetime.now().isoformat()
        
        # Create result object with task attributes
        result_obj = {
            "task_attributes": self.task_attributes,
            "results": results
        }
        
        # Save to file
        result_path = os.path.join(self.result_dir, f"{mission}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_obj, f, ensure_ascii=False, indent=4)
        
        self.logger.info(f"Results saved to {result_path}")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update task metrics.
        
        Args:
            metrics: Metrics to update
        """
        self.task_attributes["metrics"].update(metrics)
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Run the task.
        
        Returns:
            Task results
        """
        pass
