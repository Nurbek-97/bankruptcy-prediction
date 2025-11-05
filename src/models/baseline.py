"""Baseline models for bankruptcy prediction."""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib


class BaselineModels:
    """Baseline model implementations."""
    
    def __init__(self, config):
        """
        Initialize baseline models.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = {}
        self.trained_models = {}
        
    def create_logistic_regression(self, class_weight: str = 'balanced') -> LogisticRegression:
        """
        Create logistic regression model.
        
        Args:
            class_weight: Class weighting strategy
            
        Returns:
            Logistic regression model
        """
        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=self.config.random_seed,
            n_jobs=-1,
            solver='saga'
        )
        return model
    
    def create_decision_tree(self, max_depth: int = 10) -> DecisionTreeClassifier:
        """
        Create decision tree model.
        
        Args:
            max_depth: Maximum tree depth
            
        Returns:
            Decision tree model
        """
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight='balanced',
            random_state=self.config.random_seed,
            min_samples_split=10,
            min_samples_leaf=5
        )
        return model
    
    def create_random_forest(
        self, 
        n_estimators: int = 100,
        max_depth: int = 10
    ) -> RandomForestClassifier:
        """
        Create random forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Random forest model
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=self.config.random_seed,
            n_jobs=-1,
            min_samples_split=10,
            min_samples_leaf=5
        )
        return model
    
    def create_naive_bayes(self) -> GaussianNB:
        """
        Create Gaussian Naive Bayes model.
        
        Returns:
            Naive Bayes model
        """
        model = GaussianNB()
        return model
    
    def create_svm(self, kernel: str = 'rbf') -> SVC:
        """
        Create SVM model.
        
        Args:
            kernel: Kernel type
            
        Returns:
            SVM model
        """
        model = SVC(
            kernel=kernel,
            class_weight='balanced',
            probability=True,
            random_state=self.config.random_seed
        )
        return model
    
    def train_all_baselines(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> dict:
        """
        Train all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training baseline models...")
        
        self.models = {
            'logistic_regression': self.create_logistic_regression(),
            'decision_tree': self.create_decision_tree(),
            'random_forest': self.create_random_forest(),
            'naive_bayes': self.create_naive_bayes()
        }
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Log training score
                train_score = model.score(X_train, y_train)
                logger.info(f"{name} training accuracy: {train_score:.4f}")
                
                if X_val is not None and y_val is not None:
                    val_score = model.score(X_val, y_val)
                    logger.info(f"{name} validation accuracy: {val_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                
        logger.info("Baseline models training completed")
        
        return self.trained_models
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with a baseline model.
        
        Args:
            model_name: Name of the model
            X: Features
            
        Returns:
            Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
            
        return self.trained_models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of the model
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
            
        return self.trained_models[model_name].predict_proba(X)
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model
            filepath: Path to save model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
            
        joblib.dump(self.trained_models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_name: Name of the model
            filepath: Path to load model from
        """
        self.trained_models[model_name] = joblib.load(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")