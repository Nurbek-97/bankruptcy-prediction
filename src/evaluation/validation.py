"""Cross-validation and model validation strategies."""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, cross_val_score, cross_validate
)
from typing import Dict, List


class CrossValidator:
    """Cross-validation strategies for model evaluation."""
    
    def __init__(self, config):
        """
        Initialize cross-validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.cv_folds = config.get('models.cv_folds', 5)
        self.cv_strategy = config.get('models.cv_strategy', 'stratified')
        
    def get_cv_splitter(self):
        """
        Get cross-validation splitter.
        
        Returns:
            CV splitter object
        """
        if self.cv_strategy == 'stratified':
            return StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.config.random_seed
            )
        elif self.cv_strategy == 'time_series':
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def cross_validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: List[str] = None
    ) -> Dict:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            scoring: List of metrics to compute
            
        Returns:
            Dictionary of CV results
        """
        scoring = scoring or ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        logger.info(f"Running {self.cv_folds}-fold cross-validation...")
        
        cv_splitter = self.get_cv_splitter()
        
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate mean and std for each metric
        results_summary = {}
        
        for metric in scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            if test_key in cv_results:
                results_summary[f'{metric}_mean'] = cv_results[test_key].mean()
                results_summary[f'{metric}_std'] = cv_results[test_key].std()
                results_summary[f'{metric}_train_mean'] = cv_results[train_key].mean()
                
        logger.info("Cross-validation completed")
        
        return results_summary
    
    def compare_models_cv(
        self,
        models: Dict,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = 'roc_auc'
    ) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.
        
        Args:
            models: Dictionary of models
            X: Features
            y: Target
            scoring: Metric for comparison
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models using CV...")
        
        cv_splitter = self.get_cv_splitter()
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1
            )
            
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
            
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('mean', ascending=False)
        
        logger.info(f"\nCV Comparison Results:\n{comparison_df.to_string()}")
        
        return comparison_df
    
    def validate_temporal_stability(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        time_col: str = 'year'
    ) -> Dict:
        """
        Validate model stability across time periods.
        
        Args:
            model: Trained model
            X: Features with time column
            y: Target
            time_col: Name of time column
            
        Returns:
            Dictionary of temporal validation results
        """
        if time_col not in X.columns:
            logger.warning(f"Time column '{time_col}' not found")
            return {}
            
        logger.info("Validating temporal stability...")
        
        X_temp = X.copy()
        time_periods = sorted(X_temp[time_col].unique())
        
        results = {}
        
        for period in time_periods:
            mask = X_temp[time_col] == period
            X_period = X_temp[mask].drop(columns=[time_col])
            y_period = y[mask]
            
            if len(y_period) > 0:
                score = model.score(X_period, y_period)
                results[f'period_{period}'] = score
                logger.info(f"Period {period} score: {score:.4f}")
                
        return results