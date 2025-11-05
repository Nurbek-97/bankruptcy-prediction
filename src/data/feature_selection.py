"""Feature selection methods for bankruptcy prediction."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    VarianceThreshold,
    f_classif
)


class FeatureSelector:
    """Select important features for modeling."""
    
    def __init__(self, config):
        """
        Initialize feature selector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.selected_features = None
        
    def remove_low_variance(
        self, 
        X: pd.DataFrame,
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Remove features with low variance.
        
        Args:
            X: Feature DataFrame
            threshold: Variance threshold
            
        Returns:
            DataFrame with low variance features removed
        """
        threshold = threshold or self.config.get('preprocessing.variance_threshold', 0.01)
        
        logger.info(f"Removing features with variance < {threshold}...")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_cols = X.columns[selector.get_support()].tolist()
        removed_cols = set(X.columns) - set(selected_cols)
        
        logger.info(f"Removed {len(removed_cols)} low variance features")
        
        return X[selected_cols]
    
    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            DataFrame with correlated features removed
        """
        threshold = threshold or self.config.get('preprocessing.correlation_threshold', 0.95)
        
        logger.info(f"Removing features with correlation > {threshold}...")
        
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > threshold)
        ]
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return X.drop(columns=to_drop)
    
    def select_k_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 50,
        score_func: str = 'f_classif'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select K best features using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            k: Number of features to select
            score_func: Scoring function ('f_classif', 'mutual_info')
            
        Returns:
            Tuple of (selected DataFrame, feature names)
        """
        logger.info(f"Selecting {k} best features using {score_func}...")
        
        if score_func == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif score_func == 'mutual_info':
            selector = SelectKBest(
                score_func=lambda X, y: mutual_info_classif(
                    X, y, random_state=self.config.random_seed
                ),
                k=k
            )
        else:
            raise ValueError(f"Unknown score function: {score_func}")
            
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        logger.info(f"Top 10 features:\n{scores.head(10)}")
        
        return X[selected_features], selected_features
    
    def select_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 50,
        method: str = 'random_forest'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features by model-based importance.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            method: Model type ('random_forest')
            
        Returns:
            Tuple of (selected DataFrame, importance DataFrame)
        """
        logger.info(f"Selecting {n_features} features by {method} importance...")
        
        if method == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_seed,
                n_jobs=-1
            )
            model.fit(X, y)
            
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importances.head(n_features)['feature'].tolist()
            
            logger.info(f"Top 10 important features:\n{importances.head(10)}")
            
            return X[selected_features], importances
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        method: str = 'all'
    ) -> pd.DataFrame:
        """
        Apply multiple feature selection methods.
        
        Args:
            X: Feature DataFrame
            y: Target variable (required for supervised methods)
            method: Selection method ('variance', 'correlation', 'importance', 'all')
            
        Returns:
            DataFrame with selected features
        """
        X_selected = X.copy()
        
        if method in ['variance', 'all']:
            X_selected = self.remove_low_variance(X_selected)
            
        if method in ['correlation', 'all']:
            X_selected = self.remove_correlated_features(X_selected)
            
        if method in ['importance', 'all'] and y is not None:
            X_selected, _ = self.select_by_importance(
                X_selected, 
                y, 
                n_features=min(50, len(X_selected.columns))
            )
            
        self.selected_features = X_selected.columns.tolist()
        
        logger.info(f"Feature selection completed. Selected {len(self.selected_features)} features")
        
        return X_selected
    
    def get_selected_features(self) -> List[str]:
        """
        Get list of selected features.
        
        Returns:
            List of feature names
        """
        if self.selected_features is None:
            raise ValueError("No features selected yet. Run select_features() first.")
            
        return self.selected_features