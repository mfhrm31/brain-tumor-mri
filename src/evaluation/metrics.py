"""
Multi-class evaluation metrics for brain tumor classification.

Computes per-class and aggregate metrics, plus confusion matrix
analysis for clinically interpretable performance breakdown.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)
from typing import Dict, List, Optional


CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]


class MultiClassMetrics:
    """
    Multi-class evaluation for brain tumor classification.

    Reports:
        - Overall accuracy and Cohen's kappa
        - Per-class precision, recall, F1
        - Macro and weighted averages
        - Confusion matrix
        - Top-class confusions (most common error pairs)
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or CLASS_NAMES
        self.results_ = None

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Compute all metrics.

        Args:
            y_true: Ground-truth class labels (integers)
            y_pred: Predicted class labels (integers)

        Returns:
            Dictionary with all metric values
        """
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall_macro = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        f1_macro = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        precision_weighted = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        recall_weighted = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        f1_weighted = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred)
        top_confusions = self._top_confusions(cm)

        results = {
            "accuracy": float(accuracy),
            "cohen_kappa": float(kappa),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "class_names": list(self.class_names),
            "confusion_matrix": cm.tolist(),
            "top_confusions": top_confusions,
        }
        self.results_ = results
        return results

    def _top_confusions(self, cm: np.ndarray, top_n: int = 3) -> List[Dict]:
        """
        Identify the most common misclassification pairs.

        Args:
            cm: Confusion matrix
            top_n: Number of confusion pairs to return

        Returns:
            List of dicts describing top confusions
        """
        n = cm.shape[0]
        confusions = []
        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j] > 0:
                    confusions.append({
                        "true_class": self.class_names[i],
                        "predicted_class": self.class_names[j],
                        "count": int(cm[i, j]),
                    })
        confusions.sort(key=lambda c: c["count"], reverse=True)
        return confusions[:top_n]

    def summary_table(self) -> str:
        """Return formatted summary of metrics."""
        if self.results_ is None:
            return "No results computed. Call compute() first."

        r = self.results_
        lines = [
            "=" * 60,
            "  Brain Tumor MRI — Multi-Class Evaluation",
            "=" * 60,
            f"  Accuracy           : {r['accuracy']:.4f}",
            f"  Cohen's Kappa      : {r['cohen_kappa']:.4f}",
            "  -- Macro averages --",
            f"  Precision (macro)  : {r['precision_macro']:.4f}",
            f"  Recall (macro)     : {r['recall_macro']:.4f}",
            f"  F1-Score (macro)   : {r['f1_macro']:.4f}",
            "  -- Weighted averages --",
            f"  Precision (weight) : {r['precision_weighted']:.4f}",
            f"  Recall (weight)    : {r['recall_weighted']:.4f}",
            f"  F1-Score (weight)  : {r['f1_weighted']:.4f}",
            "  -- Per-class F1 --",
        ]
        for name, f1 in zip(r["class_names"], r["f1_per_class"]):
            lines.append(f"    {name:14s}: {f1:.4f}")

        if r["top_confusions"]:
            lines.append("  -- Top misclassifications --")
            for c in r["top_confusions"]:
                lines.append(
                    f"    {c['true_class']:12s} → "
                    f"{c['predicted_class']:12s}: {c['count']} samples"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def print_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Print full per-class sklearn classification report."""
        print(classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        ))


if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 800
    y_true = np.random.choice([0, 1, 2, 3], size=n_samples)

    y_pred = y_true.copy()
    flip_mask = np.random.rand(n_samples) < 0.12
    y_pred[flip_mask] = np.random.choice([0, 1, 2, 3], size=flip_mask.sum())

    metrics = MultiClassMetrics()
    metrics.compute(y_true, y_pred)
    print(metrics.summary_table())
