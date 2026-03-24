import os
import json
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from tasks.base_task import BaseTask
from utils import encode_image
from call import APIManager

class VideoQualityTask(BaseTask):
    """
    Task for evaluating video quality.
    """
    
    def __init__(self, model_name: str = "gpt-4o", 
                 data_path: str = "./info/video_quality_pairs_tests.json",
                 data_root: str = "/DATA/wfr/Q-eval/dataset/AGI-Eval/Q-Eval-100K/"):
        """
        Initialize the video quality task.
        
        Args:
            model_name: Name of the model being evaluated
            data_path: Path to the test data JSON file
            data_root: Root directory for dataset
        """
        super().__init__(model_name, "video_quality", data_root)
        self.data_path = data_path
        self.api_manager = APIManager()
        
    def run_sbs_evaluation(self) -> Dict[str, Any]:
        """
        Run side-by-side evaluation for video quality.
        
        Returns:
            Evaluation results
        """
        self.logger.info("Starting Video Quality Side-by-Side evaluation...")
        
        # Load test data
        with open(self.data_path) as f:
            json_data = json.load(f)
        
        cnt_acc = 0
        cnt_align = 0
        results = []
        count = 0
        
        for item in tqdm(json_data, desc="Processing items"):
            count += 1
            prompt = item[0].get("prompt", "")
            gt = True if item[0]['gt_score'] > item[1]['gt_score'] else False
            acc_flag = False
            
            # First query
            gpt_prompt = item[2]["sbs"]["question"]
            
            # Encode frames
            image1 = encode_image(os.path.join(self.data_root, item[0]['frames'][0]))
            image2 = encode_image(os.path.join(self.data_root, item[0]['frames'][2]))
            image3 = encode_image(os.path.join(self.data_root, item[0]['frames'][4]))
            image4 = encode_image(os.path.join(self.data_root, item[0]['frames'][7]))
            image5 = encode_image(os.path.join(self.data_root, item[1]['frames'][0]))
            image6 = encode_image(os.path.join(self.data_root, item[1]['frames'][2]))
            image7 = encode_image(os.path.join(self.data_root, item[1]['frames'][4]))
            image8 = encode_image(os.path.join(self.data_root, item[1]['frames'][7]))
            
            base64_frames = [image1, image2, image3, image4, image5, image6, image7, image8]
            
            # First evaluation
            pred1 = self.api_manager.call_gpt_with_images(
                gpt_prompt, 
                base64_frames, 
                "video_quality", 
                self.logger,
                model=self.model_name
            )
            
            self.logger.info(f"video quality sbs: {count}; answer: {pred1}")
            
            if 'first' in pred1.lower():
                acc_flag = gt
                flag1 = True
            else:
                acc_flag = not gt
                flag1 = False
            
            # Second evaluation with swapped videos
            base64_frames_inverse = base64_frames[4:] + base64_frames[:4]
            pred2 = self.api_manager.call_gpt_with_images(
                gpt_prompt, 
                base64_frames_inverse, 
                "video_quality", 
                self.logger,
                model=self.model_name
            )
            
            self.logger.info(f"video quality sbs: {count}; answer: {pred2}")
            
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
        
        self.logger.info(f"Video Quality SBS Acc: {acc}, Align: {align_ratio}")
        
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
        self.save_results(final_results, "video_quality_sbs")
        
        return final_results
    
    def run_choice_evaluation(self) -> Dict[str, Any]:
        """
        Run choice-based evaluation for video quality.
        
        Returns:
            Evaluation results
        """
        self.logger.info("Starting Video Quality Choice evaluation...")
        
        # Load test data
        with open(self.data_path) as f:
            json_data = json.load(f)
        
        acc_single_sum = 0
        acc_multiple_sum = 0
        precision_sum = 0
        recall_sum = 0
        results = []
        count = 0
        
        for item in tqdm(json_data, desc="Processing items"):
            count += 1
            prompt = item[0].get('prompt', '')
            score1 = item[0]['gt_score']
            score2 = item[1]['gt_score']
            
            # Determine score info
            if score1 > score2:
                score_info = f"Video 1 has a higher quality score than Video 2."
            else:
                score_info = f"Video 2 has a higher quality score than Video 1."
            
            # Encode frames
            image1 = encode_image(os.path.join(self.data_root, item[0]['frames'][0]))
            image2 = encode_image(os.path.join(self.data_root, item[0]['frames'][2]))
            image3 = encode_image(os.path.join(self.data_root, item[0]['frames'][4]))
            image4 = encode_image(os.path.join(self.data_root, item[0]['frames'][7]))
            image5 = encode_image(os.path.join(self.data_root, item[1]['frames'][0]))
            image6 = encode_image(os.path.join(self.data_root, item[1]['frames'][2]))
            image7 = encode_image(os.path.join(self.data_root, item[1]['frames'][4]))
            image8 = encode_image(os.path.join(self.data_root, item[1]['frames'][7]))
            
            base64_frames = [image1, image2, image3, image4, image5, image6, image7, image8]
            
            item_result = {
                "choices": []
            }
            
            for idx, choice in enumerate(item[2]['choices']):
                question = choice['question']
                options = choice["options"]
                answer = choice['correct']
                
                if idx == 0:
                    # Single-choice question
                    # f"{score_info}\n"
                    gpt_prompt = (
                        f"These are two videos (Each video is represented by 4 sampled frames: the first video<image_1><image_2><image_3><image_4> and the second video<image_5><image_6><image_7><image_8>) generated by different T2V models.\n"

                        f"Please answer the following single-choice question based on the comparison of the two videos.\n\n"
                        f"Question: {question}\n\n"
                        f"Options:\n"
                        + "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)]) +
                        f"\n\nPlease select the most appropriate option and only output the option number. Do not include any explanation, reasoning, or extra words."
                    )
                else:
                    # Multiple-choice question
                    gpt_prompt = (
                        f"These are two videos (Each video is represented by 4 sampled frames: the first video<image_1><image_2><image_3><image_4> and the second video<image_5><image_6><image_7><image_8>) generated by different T2V models.\n"
                        f"Please answer the following multiple-choice question based on the comparison of the two videos.\n\n"
                        f"Question: {question}\n\n"
                        f"Options:\n"
                        + "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)]) +
                        f"\n\nPlease select all appropriate options and only output the option numbers separated by commas (e.g., 0,1,3). Do not include any explanation, reasoning, or extra words."
                    )
                
                # Call API
                pred_text = self.api_manager.call_gpt_with_images(
                    gpt_prompt, 
                    base64_frames, 
                    "video_quality_choice", 
                    self.logger,
                    model=self.model_name
                )
                
                self.logger.info(f"video quality choice: {count}; answer: {pred_text}")
                
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
        self.save_results(final_results, "video_quality_choice")
        
        return final_results
    
    def run(self, evaluation_type: str = "both") -> Dict[str, Any]:
        """
        Run the video quality task.
        
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
