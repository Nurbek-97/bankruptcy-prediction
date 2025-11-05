"""Performance metrics and evaluation for bankruptcy prediction."""

import matplotlib
matplotlib.use('Agg')  # CRITICAL: Set non-interactive backend BEFORE importing pyplot

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    brier_score_loss, matthews_corrcoef, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelMetrics:
    """Calculate and visualize model performance metrics."""
    
    def __init__(self, config):
        """
        Initialize metrics calculator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.figures_path = Path(config.get('output.figures_path'))
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> dict:
        """
        Calculate comprehensive set of metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating performance metrics...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            # Handle both formats: (n_samples,) and (n_samples, 2)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                proba = y_pred_proba[:, 1]
            else:
                proba = y_pred_proba
                
            metrics['roc_auc'] = roc_auc_score(y_true, proba)
            metrics['pr_auc'] = average_precision_score(y_true, proba)
            metrics['brier_score'] = brier_score_loss(y_true, proba)
            
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        logger.info("Metrics calculation completed")
        
        return metrics
    
    def print_metrics(self, metrics: dict, model_name: str = "Model") -> None:
        """
        Print metrics in formatted table.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"{model_name} Performance Metrics")
        logger.info(f"{'='*50}")
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric_name:.<30} {value:.4f}")
            else:
                logger.info(f"{metric_name:.<30} {value}")
                
        logger.info(f"{'='*50}\n")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name for title
            save: Whether to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Non-Bankruptcy', 'Bankruptcy'],
            yticklabels=['Non-Bankruptcy', 'Bankruptcy']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save:
            filepath = self.figures_path / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {filepath}")
            
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Model name for title
            save: Whether to save figure
        """
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            proba = y_pred_proba[:, 1]
        else:
            proba = y_pred_proba
            
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        roc_auc = roc_auc_score(y_true, proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = self.figures_path / f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {filepath}")
            
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> None:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Model name for title
            save: Whether to save figure
        """
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            proba = y_pred_proba[:, 1]
        else:
            proba = y_pred_proba
            
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        pr_auc = average_precision_score(y_true, proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = self.figures_path / f'pr_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {filepath}")
            
        plt.close()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str = "Model",
        top_n: int = 20,
        save: bool = True
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            model_name: Model name for title
            top_n: Number of top features to plot
            save: Whether to save figure
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save:
            filepath = self.figures_path / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {filepath}")
            
        plt.close()
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> float:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            proba = y_pred_proba[:, 1]
        else:
            proba = y_pred_proba
            
        thresholds = np.linspace(0.1, 0.9, 100)
        scores = []
        
        for threshold in thresholds:
            y_pred = (proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            scores.append(score)
            
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.4f} (score: {optimal_score:.4f})")
        
        return optimal_threshold
    
    def compare_models(
        self,
        results: dict,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results: Dictionary of {model_name: metrics_dict}
            save: Whether to save comparison
            
        Returns:
            Comparison DataFrame
        """
        logger.info("Comparing models...")
        
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        
        # Sort by primary metric
        primary_metric = self.config.get('evaluation.primary_metric', 'roc_auc')
        if primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
            
        logger.info(f"\nModel Comparison:\n{comparison_df.to_string()}")
        
        if save:
            results_path = Path(self.config.get('output.results_path'))
            filepath = results_path / 'model_comparison.csv'
            comparison_df.to_csv(filepath)
            logger.info(f"Model comparison saved to {filepath}")
            
        return comparison_df
    
    def save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> None:
        """
        Save detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name
        """
        report = classification_report(
            y_true, 
            y_pred,
            target_names=['Non-Bankruptcy', 'Bankruptcy'],
            digits=4
        )
        
        logger.info(f"\nClassification Report - {model_name}:\n{report}")
        
        results_path = Path(self.config.get('output.results_path'))
        filepath = results_path / f'classification_report_{model_name.lower().replace(" ", "_")}.txt'
        
        with open(filepath, 'w') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("="*50 + "\n\n")
            f.write(report)
            
        logger.info(f"Classification report saved to {filepath}")