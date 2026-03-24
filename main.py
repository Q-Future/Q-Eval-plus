#!/usr/bin/env python3
"""
Main entry point for running Q-Eval evaluations.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional

from tasks.base_task import BaseTask
from tasks.image_alignment import ImageAlignmentTask
from tasks.image_quality import ImageQualityTask
from tasks.video_alignment import VideoAlignmentTask
from tasks.video_quality import VideoQualityTask
from call import APIManager

def setup_logging():
    """
    Set up logging for the main script.
    
    Returns:
        Configured logger
    """
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "main.log")
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
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

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run Q-Eval evaluations")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o", 
        help="Model to evaluate"
    )
    
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["image_alignment", "image_quality", "video_alignment", "video_quality", "all"], 
        default="all", 
        help="Task to run"
    )
    
    parser.add_argument(
        "--evaluation_type", 
        type=str, 
        choices=["sbs", "choice", "both"], 
        default="both", 
        help="Type of evaluation to run"
    )
    
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="/path/to/dataset/AGI-Eval/Q-Eval-100K/", 
        help="Root directory for dataset, Download link: https://huggingface.co/datasets/meituan-longcat/Q-Eval-100K/ "
    )
    
    parser.add_argument(
        "--image_alignment_data", 
        type=str, 
        default="./info/image_alignment_pairs_test_part_summary_choices.json", 
        help="Path to image alignment test data"
    )
    
    parser.add_argument(
        "--image_quality_data", 
        type=str, 
        default="./info/image_quality_pairs_test_part_summary_choice.json", 
        help="Path to image quality test data"
    )
    
    parser.add_argument(
        "--video_alignment_data", 
        type=str, 
        default="./info/video_alignment_pairs_test_part_summary_choices.json", 
        help="Path to video alignment test data"
    )
    
    parser.add_argument(
        "--video_quality_data", 
        type=str, 
        default="./info/video_quality_pairs_test_part_summary_choices.json", 
        help="Path to video quality test data"
    )
    
    return parser.parse_args()

def run_task(task_name: str, model: str, evaluation_type: str, data_path: str, data_root: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Run a specific evaluation task.
    
    Args:
        task_name: Name of the task to run
        model: Model to evaluate
        evaluation_type: Type of evaluation to run
        data_path: Path to test data
        data_root: Root directory for dataset
        logger: Logger to use
        
    Returns:
        Task results
    """
    logger.info(f"Running {task_name} task with model {model}")
    
    if task_name == "image_alignment":
        task = ImageAlignmentTask(model, data_path, data_root)
    elif task_name == "image_quality":
        task = ImageQualityTask(model, data_path, data_root)
    elif task_name == "video_alignment":
        task = VideoAlignmentTask(model, data_path, data_root)
    elif task_name == "video_quality":
        task = VideoQualityTask(model, data_path, data_root)
    else:
        logger.error(f"Unknown task: {task_name}")
        return {}
    
    results = task.run(evaluation_type)
    logger.info(f"Completed {task_name} task")
    
    return results

def main():
    """
    Main function.
    """
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Initialize API manager to ensure config file exists
    api_manager = APIManager()
    
    # Run tasks
    if args.task == "all":
        # Run all tasks
        tasks = [
            ("image_alignment", args.image_alignment_data),
            ("image_quality", args.image_quality_data),
            ("video_alignment", args.video_alignment_data),
            ("video_quality", args.video_quality_data)
        ]
        
        for task_name, data_path in tasks:
            run_task(task_name, args.model, args.evaluation_type, data_path, args.data_root, logger)
    else:
        # Run specific task
        data_path = getattr(args, f"{args.task}_data")
        run_task(args.task, args.model, args.evaluation_type, data_path, args.data_root, logger)
    
    logger.info("All tasks completed")

if __name__ == "__main__":
    main()
