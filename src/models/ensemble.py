"""Ensemble methods for bankruptcy prediction."""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier
import joblib


class EnsembleModel:
    """Ensemble model implementations."""
    
    def __init__(self, config):
        """
        Initialize ensemble model.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.ensemble_model = None
        self.base_models = {}
        
    def create_stacking_ensemble(
        self,
        base_models: dict,
        meta_model: str = None
    ) -> StackingClassifier:
        """
        Create stacking ensemble.
        
        Args:
            base_models: Dictionary of base models
            meta_model: Type of meta-learner
            
        Returns:
            Stacking classifier
        """
        meta_model = meta_model or self.config.get('ensemble.meta_model', 'logistic_regression')
        
        logger.info(f"Creating stacking ensemble with {len(base_models)} base models")
        
        # Prepare estimators list
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Create meta-learner
        if meta_model == 'logistic_regression':
            final_estimator = LogisticRegression(
                class_weight='balanced',
                random_state=self.config.random_seed
            )
        else:
            raise ValueError(f"Unknown meta model: {meta_model}")
            
        # Create stacking classifier
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            stack_method='predict_proba' if self.config.get('ensemble.use_probabilities', True) else 'predict',
            n_jobs=-1
        )
        
        return stacking
    
    def create_voting_ensemble(
        self,
        base_models: dict,
        voting: str = 'soft',
        weights: list = None
    ) -> VotingClassifier:
        """
        Create voting ensemble.
        
        Args:
            base_models: Dictionary of base models
            voting: Voting type ('hard' or 'soft')
            weights: Model weights
            
        Returns:
            Voting classifier
        """
        logger.info(f"Creating {voting} voting ensemble with {len(base_models)} base models")
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
        return voting_clf
    
    def create_weighted_average_ensemble(
        self,
        base_models: dict,
        weights: dict = None
    ) -> None:
        """
        Create weighted average ensemble (custom implementation).
        
        Args:
            base_models: Dictionary of trained base models
            weights: Model weights dictionary
        """
        weights = weights or self.config.get('ensemble.model_weights', {})
        
        logger.info(f"Creating weighted average ensemble")
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        self.base_models = base_models
        self.weights = normalized_weights
        
        logger.info(f"Ensemble weights: {normalized_weights}")
    
    def train_stacking_ensemble(
        self,
        base_models: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> StackingClassifier:
        """
        Train stacking ensemble.
        
        Args:
            base_models: Dictionary of base models
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained stacking model
        """
        logger.info("Training stacking ensemble...")
        
        self.ensemble_model = self.create_stacking_ensemble(base_models)
        self.ensemble_model.fit(X_train, y_train)
        
        logger.info("Stacking ensemble training completed")
        
        return self.ensemble_model
    
    def train_voting_ensemble(
        self,
        base_models: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        voting: str = 'soft',
        weights: list = None
    ) -> VotingClassifier:
        """
        Train voting ensemble.
        
        Args:
            base_models: Dictionary of base models
            X_train: Training features
            y_train: Training labels
            voting: Voting type
            weights: Model weights
            
        Returns:
            Trained voting model
        """
        logger.info("Training voting ensemble...")
        
        self.ensemble_model = self.create_voting_ensemble(base_models, voting, weights)
        self.ensemble_model.fit(X_train, y_train)
        
        logger.info("Voting ensemble training completed")
        
        return self.ensemble_model
    
    def predict_weighted_average(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using weighted average.
        
        Args:
            X: Features
            
        Returns:
            Ensemble predictions
        """
        if not self.base_models:
            raise ValueError("Base models not set")
            
        predictions = []
        
        for model_name, model in self.base_models.items():
            if model_name in self.weights:
                proba = model.predict_proba(X)[:, 1]
                weighted_proba = proba * self.weights[model_name]
                predictions.append(weighted_proba)
                
        ensemble_proba = np.sum(predictions, axis=0)
        
        return (ensemble_proba >= 0.5).astype(int)
    
    def predict_proba_weighted_average(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble prediction probabilities using weighted average.
        
        Args:
            X: Features
            
        Returns:
            Ensemble probabilities
        """
        if not self.base_models:
            raise ValueError("Base models not set")
            
        predictions = []
        
        for model_name, model in self.base_models.items():
            if model_name in self.weights:
                proba = model.predict_proba(X)[:, 1]
                weighted_proba = proba * self.weights[model_name]
                predictions.append(weighted_proba)
                
        ensemble_proba = np.sum(predictions, axis=0)
        
        return np.vstack([1 - ensemble_proba, ensemble_proba]).T
    
    def predict(self, X: pd.DataFrame, method: str = 'stacking') -> np.ndarray:
        """
        Make predictions using ensemble.
        
        Args:
            X: Features
            method: Ensemble method ('stacking', 'voting', 'weighted')
            
        Returns:
            Predictions
        """
        if method == 'weighted':
            return self.predict_weighted_average(X)
        elif self.ensemble_model is not None:
            return self.ensemble_model.predict(X)
        else:
            raise ValueError("Ensemble model not trained")
    
    def predict_proba(self, X: pd.DataFrame, method: str = 'stacking') -> np.ndarray:
        """
        Get ensemble prediction probabilities.
        
        Args:
            X: Features
            method: Ensemble method
            
        Returns:
            Prediction probabilities
        """
        if method == 'weighted':
            return self.predict_proba_weighted_average(X)
        elif self.ensemble_model is not None:
            return self.ensemble_model.predict_proba(X)
        else:
            raise ValueError("Ensemble model not trained")
    
    def save_ensemble(self, filepath: str) -> None:
        """
        Save ensemble model.
        
        Args:
            filepath: Path to save model
        """
        if self.ensemble_model is not None:
            joblib.dump(self.ensemble_model, filepath)
        else:
            joblib.dump({
                'base_models': self.base_models,
                'weights': self.weights
            }, filepath)
            
        logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str) -> None:
        """
        Load ensemble model.
        
        Args:
            filepath: Path to load model from
        """
        loaded = joblib.load(filepath)
        
        if isinstance(loaded, dict):
            self.base_models = loaded['base_models']
            self.weights = loaded['weights']
        else:
            self.ensemble_model = loaded
            
        logger.info(f"Ensemble loaded from {filepath}")