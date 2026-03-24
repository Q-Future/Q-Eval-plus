from openai import OpenAI
import time
import json
import logging
from typing import Dict, Any, List, Optional
from utils import get_sleep_interval

class APIManager:
    """
    Manages API calls to different models and endpoints.
    """
    
    def __init__(self, config_path: str = "./api_config.json"):
        """
        Initialize the API manager.
        
        Args:
            config_path: Path to the API configuration file
        """
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        """
        Load API configuration from the config file.
        """
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"API config loading ERROR: {e}")
            
    
    def save_config(self):
        """
        Save the current configuration to the config file.
        """
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_client(self, task_type: str = "default"):
        """
        Get an OpenAI client for the specified task type.
        
        Args:
            task_type: Type of task (e.g., "image_align", "video_quality")
            
        Returns:
            OpenAI client configured for the specified task
        """
        if task_type not in self.config:
            task_type = "default"
            
        config = self.config[task_type]
        return OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"]
        )
    
    def call_gpt(self, content: List[Dict[str, Any]], task_type: str = "default", 
                model: str = "gpt-4o", max_retries: int = 10, 
                retry_delay: int = 5, logger: Optional[logging.Logger] = None):
        """
        Call GPT model with the given content.
        
        Args:
            content: List of content items to send to the model
            task_type: Type of task (e.g., "image_align", "video_quality")
            model: Model to use
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            logger: Logger to use for logging
            
        Returns:
            Model response or "None" if all retries fail
        """
        client = self.get_client(task_type)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                )
                pred = response.choices[0].message.content
                if logger:
                    logger.info(f"API response: {pred}")
                return pred
            except Exception as e:
                retry_count += 1
                if logger:
                    logger.warning(f"API call failed (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count >= max_retries:
                    if logger:
                        logger.error(f"All retries failed for task type: {task_type}")
                    return "None"
                time.sleep(retry_delay)
    
    def call_gpt_with_images(self, prompt: str, base64_frames: List[str], 
                           task_type: str = "default", logger: Optional[logging.Logger] = None,
                           model: str = "gpt-4o"):
        """
        Call GPT model with text prompt and images.
        
        Args:
            prompt: Text prompt
            base64_frames: List of base64-encoded images
            task_type: Type of task
            logger: Logger to use for logging
            model: Model to use for the API call
            
        Returns:
            Model response
        """
        content = [{"type": "text", "text": prompt}]
        
        for i, base64_frame in enumerate(base64_frames):
            # Add image identifier
            content.append({
                "type": "text",
                "text": f"<image_{i+1}>"
            })
            # Add image
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_frame}",
                }
            })
        
        return self.call_gpt(content, task_type, model=model, logger=logger)
    
    def call_gpt_vision(self, raw_request: Dict[str, Any], logger: Optional[logging.Logger] = None, 
                       model: str = "gpt-4o"):
        """
        Call GPT Vision with the given request.
        
        Args:
            raw_request: Request data
            logger: Logger to use for logging
            model: Model to use for the API call
            
        Returns:
            Response data
        """
        prompt = raw_request["prompt"]
        if len(prompt) == 0:
            if logger:
                logger.warning("Prompt length is 0")
        
        content = [{"type": "text", "text": prompt}]
        
        # Add images if present
        if raw_request.get("url"):
            if isinstance(raw_request["url"], dict):
                for key, value in raw_request["url"].items():
                    content.append({
                        "type": "text",
                        "text": key
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{value}" if not value.startswith("http") else value,
                        }
                    })
        
        # Call the API
        answer = self.call_gpt(content, "default", model=model, logger=logger)
        
        # Process the result
        completions = []
        tokens = []
        completions.append({
            "text": answer,
            "tokens": tokens,
            "logprobs": [0.0] * len(tokens),
            "top_logprobs_dicts": [{token: 0.0} for token in tokens]
        })
        
        return {"completions": completions, "input_length": len(prompt)}
