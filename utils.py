import re
import base64
import logging
from typing import List, Set

def calc_metrics(pred, answer):
    """
    Calculate evaluation metrics for multiple-choice questions.
    
    Args:
        pred: List of predicted answers
        answer: List of correct answers
        
    Returns:
        recall, precision, exact_match
    """
    pred_set = set(pred)
    answer_set = set(answer)
    # Number of correct predictions
    correct = len(pred_set & answer_set)
    # Recall
    recall = correct / len(answer_set) if answer_set else 0.0
    # Precision
    precision = correct / len(pred_set) if pred_set else 0.0
    # Exact match
    exact_match = int(pred_set == answer_set)
    return recall, precision, exact_match

def extract_answer(text):
    """
    Extract the first number from text (for single-choice questions).
    
    Args:
        text: Response text
        
    Returns:
        Extracted number or -1 if not found
    """
    match = re.search(r'\b(\d+)\b', text)
    if match:
        return int(match.group(1))
    else:
        return -1

def extract_multi_answer(text):
    """
    Extract all numbers from text (for multiple-choice questions).
    
    Args:
        text: Response text
        
    Returns:
        List of extracted numbers
    """
    result = re.findall(r'\b\d+\b', text)
    return [int(x) for x in result] if result else []

def get_sleep_interval():
    """
    Calculate sleep interval until the next minute.
    
    Returns:
        Number of seconds to sleep
    """
    from datetime import datetime, timedelta
    now = datetime.now()
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    delta = next_minute - now
    seconds = delta.seconds
    return seconds + 1
 
def encode_image(image_path):
    """
    Encode an image to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def setup_logger(log_file, name=None):
    """
    Set up a logger with file and console handlers.
    
    Args:
        log_file: Path to the log file
        name: Logger name (optional)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
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

def convert_pred(pred):
    """
    Placeholder for the convert_pred function mentioned in the original code.
    This function should be implemented based on its usage in the original code.
    """
    # Implementation depends on how it's used in the original code
    return pred
