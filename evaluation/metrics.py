import warnings
import torch
import numpy as np
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, DiceMetric, MeanIoU
from scipy.stats import sem
from scipy.stats import t as t_dist

class MetricCalculator:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        
        # MONAI metrics (handle batch processing well)
        self.hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        
    def compute_batch(self, preds, targets):
        """
        Compute metrics for a batch of predictions.
        preds: (B, C, H, W) one-hot or (B, H, W) integers
        targets: (B, H, W) integers
        """
        # Ensure preds are one-hot format for MONAI metrics
        if preds.ndim == 3:
            # If preds are (B, H, W), convert to one-hot (B, C, H, W)
            preds_onehot = torch.zeros((preds.shape[0], self.num_classes, preds.shape[1], preds.shape[2]), device=self.device)
            preds_onehot.scatter_(1, preds.unsqueeze(1), 1)
        else:
            # Assume preds are (B, C, H, W) logits or softmax
            preds_onehot = (preds == preds.max(dim=1, keepdim=True)[0]).float()
            
        # Ensure targets are one-hot format for MONAI metrics
        if targets.ndim == 3:
            targets_onehot = torch.zeros((targets.shape[0], self.num_classes, targets.shape[1], targets.shape[2]), device=self.device)
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        else:
            targets_onehot = targets

        batch_metrics = []
        
        # Convert to numpy for some simple calculations
        preds_np = preds_onehot.cpu().numpy()
        targets_np = targets_onehot.cpu().numpy()
        
        B = preds_np.shape[0]
        
        for b in range(B):
            sample_metrics = {}
            for cls in range(1, self.num_classes): # Skip background
                p = preds_np[b, cls]
                t = targets_np[b, cls]
                
                # Intersection & Union
                intersection = np.sum(p * t)
                union = np.sum(p) + np.sum(t) - intersection
                sum_p = np.sum(p)
                sum_t = np.sum(t)
                
                # Dice
                dice = (2.0 * intersection) / (sum_p + sum_t + 1e-8)
                sample_metrics[f'dice_class_{cls}'] = dice
                
                # IoU
                iou = intersection / (union + 1e-8)
                sample_metrics[f'iou_class_{cls}'] = iou
                
                # GED (Generalized Energy Distance) for deterministic
                # Approx as 2 * (1 - IoU) or similar distance measure
                # Here we use 1 - IoU as the distance base
                ged = 2 * (1 - iou)
                sample_metrics[f'ged_class_{cls}'] = ged

                # Precision/Recall
                tp = intersection
                fp = sum_p - intersection
                fn = sum_t - intersection
                
                sample_metrics[f'precision_class_{cls}'] = tp / (tp + fp + 1e-8)
                sample_metrics[f'recall_class_{cls}'] = tp / (tp + fn + 1e-8)
                
            batch_metrics.append(sample_metrics)
            
        # Compute HD95 using MONAI (it expects batches)
        # Note: HD95 can be slow and might fail if one mask is empty.
        # We process it batch-wise but MONAI handles empty masks by returning NaN or inf usually.
        try:
            self.hd95_metric(y_pred=preds_onehot, y=targets_onehot)
        except Exception as e:
            warnings.warn(f"HD95 computation failed for batch: {e}")
            
        return batch_metrics

    def aggregate_and_ci(self, all_metrics):
        """
        Aggregate metrics and compute 95% Confidence Intervals.
        all_metrics: List of dicts (one per sample)
        """
        summary = {}

        if not all_metrics:
            print("Warning: No metrics to aggregate (empty validation set)")
            self.hd95_metric.reset()
            return summary

        # Collect values lists
        keys = all_metrics[0].keys()
        data = {k: [] for k in keys}
        for m in all_metrics:
            for k, v in m.items():
                data[k].append(v)
        
        # Calculate Mean and CI
        confidence = 0.95
        for k, v in data.items():
            a = 1.0 * np.array(v)
            n = len(a)
            m, se = np.mean(a), sem(a)
            h = se * t_dist.ppf((1 + confidence) / 2., n-1)
            
            summary[k] = {
                'mean': m,
                'std': np.std(a),
                'ci_95': h,
                'min': np.min(a),
                'max': np.max(a)
            }
            
        # Get HD95 from MONAI accumulator
        # Result is (B, C) or aggregated. 
        # MONAI aggregate() returns mean usually.
        try:
            hd95_results = self.hd95_metric.aggregate() # Shape (C_foreground,) if reduction='mean'
            if isinstance(hd95_results, torch.Tensor):
                hd95_results = hd95_results.cpu().numpy()
            
            # Map back to classes (assuming include_background=False, so index 0 is class 1)
            if self.num_classes == 2:
                # Binary case
                summary['hd95_class_1'] = {'mean': float(hd95_results)}
            else:
                for i, val in enumerate(hd95_results):
                    summary[f'hd95_class_{i+1}'] = {'mean': float(val)}
                    
        except Exception as e:
            print(f"HD95 extraction failed: {e}")
            
        self.hd95_metric.reset()
        
        return summary
