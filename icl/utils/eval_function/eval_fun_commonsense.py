import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import wandb
class EvalFun:
    def __init__(self):
        pass

    def extract_a_or_b(self, text: str) -> str:
        match = re.search(r'\b(A|B)\)', text)
        return match.group(1) if match else 'none'
    
    def extract_text_after_marker(self, text, marker="</Answer>assistant"):
        if type(text) == list: text = text[0]
        text = text.replace('\n', '')
        index = text.rfind(marker)
        if index == -1:
            return ""
        return text[index + len(marker):].strip() 

    def __call__(self, response_list, gt_answers, subset_ids, marker):
        total_score = 0
        score_per_answer = defaultdict(list)
        pred_and_gt = []
        num_none = 0
        
        # Track predictions and ground truth for each class
        class_predictions = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for i in tqdm(range(len(response_list))):
            gt = gt_answers[subset_ids[i]].upper()
            response = self.extract_text_after_marker(response_list[i], marker=marker)

            if len(response) == 1: prediction = response.capitalize()
            else: prediction = self.extract_a_or_b(response.upper())

            if prediction == 'none': num_none += 1

            pred_and_gt.append((prediction, gt))
            
            # Update metrics
            try:
                if gt == prediction:
                    total_score += 1
                    score_per_answer[gt].append(1)
                    class_predictions[gt]['tp'] += 1
                else:
                    score_per_answer[gt].append(0)
                    class_predictions[gt]['fn'] += 1
                    if prediction != 'none':
                        class_predictions[prediction]['fp'] += 1
            except:
                score_per_answer[gt].append(0)
                pass
                
        print("\n##### RESULT #####")
        print("Total Score: ", total_score)
        print("Acc: ", round(100*(total_score / len(response_list)), 3))
        print("Num None: ", num_none)
        
        # Print metrics for each class
        print("\nPer-class metrics:")
        for cls, metrics in class_predictions.items():
            tp = metrics['tp']
            fp = metrics['fp'] 
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Class: {cls}")
            print(f"Precision: {round(100*precision, 3)}%")
            print(f"Recall: {round(100*recall, 3)}%") 
            print(f"F1: {round(100*f1, 3)}%")

        wandb.log({
            'total_score': total_score,
            'acc': round(100*(total_score / len(response_list)), 2),
            'num_none': num_none
        })

        return pred_and_gt