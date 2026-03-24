import os
import json
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from tasks.base_task import BaseTask
from utils import encode_image
from call import APIManager

class ImageQualityTask(BaseTask):
    """
    Task for evaluating image quality.
    """
    
    def __init__(self, model_name: str = "gpt-4o", 
                 data_path: str = "./info/image_quality_pairs_test.json",
                 data_root: str = "/DATA/wfr/Q-eval/dataset/AGI-Eval/Q-Eval-100K/"):
        """
        Initialize the image quality task.
        
        Args:
            model_name: Name of the model being evaluated
            data_path: Path to the test data JSON file
            data_root: Root directory for dataset
        """
        super().__init__(model_name, "image_quality", data_root)
        self.data_path = data_path
        self.api_manager = APIManager()
        
    def run_sbs_evaluation(self) -> Dict[str, Any]:
        """
        Run side-by-side evaluation for image quality.
        
        Returns:
            Evaluation results
        """
        self.logger.info("Starting Image Quality Side-by-Side evaluation...")
        
        # Load test data
        with open(self.data_path) as f:
            json_data = json.load(f)
        
        cnt_acc = 0
        cnt_align = 0
        results = []
        
        for item in tqdm(json_data, desc="Processing items"):
            gt = True if item[0]['gt_score'] > item[1]['gt_score'] else False
            acc_flag = False
            
            # First query
            gpt_prompt = item[2]["sbs"]["question"]
            
            image0_path = os.path.join(self.data_root, item[0]['image_path'])
            image1_path = os.path.join(self.data_root, item[1]['image_path'])
            
            image0 = encode_image(image0_path)
            image1 = encode_image(image1_path)
            
            base64_frames = [image0, image1]
            
            # First evaluation
            pred1 = self.api_manager.call_gpt_with_images(
                gpt_prompt, 
                base64_frames, 
                "image_quality", 
                self.logger,
                model=self.model_name
            )
            
            if 'first' in pred1.lower():
                acc_flag = gt
                flag1 = True
            else:
                acc_flag = not gt
                flag1 = False
            
            # Second evaluation with swapped images
            pred2 = self.api_manager.call_gpt_with_images(
                gpt_prompt, 
                base64_frames[::-1], 
                "image_quality", 
                self.logger,
                model=self.model_name
            )
            
            if acc_flag:
                if 'first' in pred2.lower():
                    acc_flag = not gt
                else:
                    acc_flag = gt
                
            flag2 = 'first' in pred2.lower()
            
            cnt_align += (flag1 != flag2)
            cnt_acc += acc_flag
            
            # Store result for this item
            item_result = {
                "gt": gt,
                "pred1": pred1,
                "pred2": pred2,
                "acc_flag": acc_flag,
                "consistency": (flag1 != flag2)
            }
            results.append(item_result)
        
        # Calculate metrics
        total_items = len(json_data)
        acc = cnt_acc / total_items
        align_ratio = cnt_align / total_items
        
        # Update task metrics
        self.update_metrics({
            "accuracy": acc,
            "alignment_ratio": align_ratio,
            "total_items": total_items
        })
        
        self.logger.info(f"Image Quality SBS Acc: {acc}, Align: {align_ratio}")
        
        # Prepare final results
        final_results = {
            "metrics": {
                "accuracy": acc,
                "alignment_ratio": align_ratio,
                "total_items": total_items
            },
            "item_results": results
        }
        
        # Save results
        self.save_results(final_results, "image_quality_sbs")
        
        return final_results
    
    def run_choice_evaluation(self) -> Dict[str, Any]:
        """
        Run choice-based evaluation for image quality.
        
        Returns:
            Evaluation results
        """
        self.logger.info("Starting Image Quality Choice evaluation...")
        
        # Load test data
        with open(self.data_path) as f:
            json_data = json.load(f)
        
        acc_single_sum = 0
        acc_multiple_sum = 0
        precision_sum = 0
        recall_sum = 0
        results = []
        
        for item in tqdm(json_data, desc="Processing items"):
            prompt = item[0].get('prompt', '')  # Prompt might not be relevant for quality
            score1 = item[0]['gt_score']
            score2 = item[1]['gt_score']
            
            image0_path = os.path.join(self.data_root, item[0]['image_path'])
            image1_path = os.path.join(self.data_root, item[1]['image_path'])
            
            image0 = encode_image(image0_path)
            image1 = encode_image(image1_path)
            
            base64_frames = [image0, image1]
            
            # Determine score info
            if score1 > score2:
                score_info = f"Image 0 has a higher quality score than Image 1."
            else:
                score_info = f"Image 1 has a higher quality score than Image 0."
            
            item_result = {
                "choices": []
            }
            
            for idx, choice in enumerate(item[2]['choices']):
                question = choice['question']
                options = choice["options"]
                answer = choice['correct']
                
                if idx == 0:
                    # Single-choice question
                    #f"{score_info}\n"
                    gpt_prompt = (
                        f"These are two images <image_0><image_1> generated from different T2I Models.\n"
                        f"Please answer the following single-choice question based on the comparison of the two images.\n\n"
                        f"Question: {question}\n\n"
                        f"Options:\n"
                        + "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)]) +
                        f"\n\nPlease select the most appropriate option and only output the option number. Do not include any explanation, reasoning, or extra words."
                    )
                else:
                    # Multiple-choice question
                    gpt_prompt = (
                        f"These are two images <image_0><image_1> generated from different T2I Models.\n"
                        f"Please answer the following multiple-choice question based on the comparison of the two images.\n\n"
                        f"Question: {question}\n\n"
                        f"Options:\n"
                        + "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)]) +
                        f"\n\nPlease select all appropriate options and only output the option numbers separated by commas (e.g., 0,1,3).Do not include any explanation, reasoning, or extra words."
                    )
                
                # Call API
                pred_text = self.api_manager.call_gpt_with_images(
                    gpt_prompt, 
                    base64_frames, 
                    "image_quality_choice", 
                    self.logger,
                    model=self.model_name
                )
                
                # Process results
                choice_result = {
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "pred_text": pred_text
                }
                
                if idx == 0:
                    # Single-choice
                    from utils import extract_answer
                    pred = extract_answer(pred_text)
                    is_correct = (pred == answer[0])
                    acc_single_sum += is_correct
                    
                    choice_result["pred"] = pred
                    choice_result["is_correct"] = is_correct
                else:
                    # Multiple-choice
                    from utils import extract_multi_answer, calc_metrics
                    pred = extract_multi_answer(pred_text)
                    recall, precision, exact_match = calc_metrics(pred, answer)
                    
                    recall_sum += recall
                    precision_sum += precision
                    acc_multiple_sum += exact_match
                    
                    choice_result["pred"] = pred
                    choice_result["recall"] = recall
                    choice_result["precision"] = precision
                    choice_result["exact_match"] = exact_match
                
                item_result["choices"].append(choice_result)
            
            results.append(item_result)
        
        # Calculate metrics
        total_items = len(json_data)
        single_choice_acc = acc_single_sum / total_items
        multiple_choice_acc = acc_multiple_sum / total_items / 2  # Assuming 2 multiple-choice questions per item
        multiple_choice_precision = precision_sum / total_items / 2
        multiple_choice_recall = recall_sum / total_items / 2
        
        # Update task metrics
        self.update_metrics({
            "single_choice_accuracy": single_choice_acc,
            "multiple_choice_accuracy": multiple_choice_acc,
            "multiple_choice_precision": multiple_choice_precision,
            "multiple_choice_recall": multiple_choice_recall,
            "total_items": total_items
        })
        
        self.logger.info(f"Single Choice Acc: {single_choice_acc}")
        self.logger.info(f"Multiple Choice Acc: {multiple_choice_acc}")
        self.logger.info(f"Multiple Choice Precision: {multiple_choice_precision}")
        self.logger.info(f"Multiple Choice Recall: {multiple_choice_recall}")
        
        # Prepare final results
        final_results = {
            "metrics": {
                "single_choice_accuracy": single_choice_acc,
                "multiple_choice_accuracy": multiple_choice_acc,
                "multiple_choice_precision": multiple_choice_precision,
                "multiple_choice_recall": multiple_choice_recall,
                "total_items": total_items
            },
            "item_results": results
        }
        
        # Save results
        self.save_results(final_results, "image_quality_choice")
        
        return final_results
    
    def run(self, evaluation_type: str = "both") -> Dict[str, Any]:
        """
        Run the image quality task.
        
        Args:
            evaluation_type: Type of evaluation to run ("sbs", "choice", or "both")
            
        Returns:
            Evaluation results
        """
        results = {}
        
        if evaluation_type in ["sbs", "both"]:
            results["sbs"] = self.run_sbs_evaluation()
        
        if evaluation_type in ["choice", "both"]:
            results["choice"] = self.run_choice_evaluation()
        
        return results
