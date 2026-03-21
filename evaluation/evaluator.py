import os
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .metrics import MetricCalculator
from .complexity import get_model_complexity
from .visualization import visualize_predictions, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


class Evaluator:
    def __init__(self, model: nn.Module, val_loader: 'DataLoader',
                 device: torch.device, config: 'TrainingConfig',
                 num_samples: int = -1, output_dir: str = 'evaluation_results'):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.num_samples = num_samples
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.metric_calc = MetricCalculator(config.NUM_CLASSES, device)

    def run(self) -> Dict:
        print("Starting evaluation...")

        # Model complexity
        input_shape = (1, self.config.NUM_CHANNELS, *self.config.IMAGE_SIZE)
        complexity = get_model_complexity(self.model, input_shape, self.device)
        print(f"Complexity: {complexity}")

        # Inference
        self.model.eval()
        all_metrics: List[Dict] = []
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                pred = outputs['pred']
                preds = torch.argmax(pred, dim=1)

                batch_metrics = self.metric_calc.compute_batch(pred, masks)
                all_metrics.extend(batch_metrics)

                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(masks.cpu().numpy().flatten())

        summary_stats = self.metric_calc.aggregate_and_ci(all_metrics)
        self.save_report(summary_stats, complexity)

        # Confusion matrix
        all_preds_arr = np.concatenate(all_preds)
        all_labels_arr = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels_arr, all_preds_arr, labels=range(self.config.NUM_CLASSES))

        class_names = self.config.get_class_names()

        plot_confusion_matrix(cm, class_names, self.output_dir)

        # Visualize
        print("Generating visualizations...")
        num_vis = len(self.val_loader.dataset) if self.num_samples == -1 else self.num_samples
        visualize_predictions(self.model, self.val_loader, self.device,
                              class_names, self.output_dir, num_samples=num_vis)

        return summary_stats

    def save_report(self, stats: Dict, complexity: Dict[str, float]) -> None:
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("       EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. MODEL COMPLEXITY\n")
            f.write("-" * 30 + "\n")
            for k, v in complexity.items():
                f.write(f"{k:<15}: {v:.4f}\n")
            f.write("\n")

            f.write("2. PERFORMANCE METRICS (95% CI)\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'CI-95%':<10}\n")
            f.write("-" * 60 + "\n")

            for k, v in stats.items():
                if isinstance(v, dict) and 'mean' in v:
                    f.write(f"{k:<20} {v['mean']:.4f}     {v.get('std', 0):.4f}     {v.get('ci_95', 0):.4f}\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"Report saved to {report_path}")
