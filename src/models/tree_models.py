"""Gradient boosting tree models for bankruptcy prediction."""

import numpy as np
import pandas as pd
from loguru import logger
import joblib
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import cross_val_score


class TreeModels:
    """Gradient boosting tree model implementations."""
    
    def __init__(self, config):
        """
        Initialize tree models.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = {}
        
    def create_xgboost(self, params: dict = None) -> xgb.XGBClassifier:
        """
        Create XGBoost model.
        
        Args:
            params: Custom parameters (optional)
            
        Returns:
            XGBoost classifier
        """
        default_params = self.config.get('models.xgboost', {})
        
        if params:
            default_params.update(params)
            
        model = xgb.XGBClassifier(
            n_estimators=default_params.get('n_estimators', 1000),
            max_depth=default_params.get('max_depth', 6),
            learning_rate=default_params.get('learning_rate', 0.01),
            subsample=default_params.get('subsample', 0.8),
            colsample_bytree=default_params.get('colsample_bytree', 0.8),
            scale_pos_weight=default_params.get('scale_pos_weight', 30),
            random_state=self.config.random_seed,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='auc',
            early_stopping_rounds=default_params.get('early_stopping_rounds', 50)
        )
        
        return model
    
    def create_lightgbm(self, params: dict = None) -> lgb.LGBMClassifier:
        """
        Create LightGBM model.
        
        Args:
            params: Custom parameters (optional)
            
        Returns:
            LightGBM classifier
        """
        default_params = self.config.get('models.lightgbm', {})
        
        if params:
            default_params.update(params)
            
        model = lgb.LGBMClassifier(
            n_estimators=default_params.get('n_estimators', 1000),
            max_depth=default_params.get('max_depth', 6),
            learning_rate=default_params.get('learning_rate', 0.01),
            num_leaves=default_params.get('num_leaves', 31),
            subsample=default_params.get('subsample', 0.8),
            colsample_bytree=default_params.get('colsample_bytree', 0.8),
            is_unbalance=default_params.get('is_unbalance', True),
            random_state=self.config.random_seed,
            n_jobs=-1,
            verbose=-1
        )
        
        return model
    
    def create_catboost(self, params: dict = None) -> cb.CatBoostClassifier:
        """
        Create CatBoost model.
        
        Args:
            params: Custom parameters (optional)
            
        Returns:
            CatBoost classifier
        """
        default_params = self.config.get('models.catboost', {})
        
        if params:
            default_params.update(params)
            
        model = cb.CatBoostClassifier(
            iterations=default_params.get('iterations', 1000),
            depth=default_params.get('depth', 6),
            learning_rate=default_params.get('learning_rate', 0.01),
            l2_leaf_reg=default_params.get('l2_leaf_reg', 3),
            subsample=default_params.get('subsample', 0.8),
            auto_class_weights=default_params.get('auto_class_weights', 'Balanced'),
            random_state=self.config.random_seed,
            verbose=False,
            thread_count=-1
        )
        
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        model = self.create_xgboost()
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            
            # Check if best_iteration exists
            try:
                best_iter = model.best_iteration
                logger.info(f"XGBoost best iteration: {best_iter}")
            except AttributeError:
                logger.info(f"XGBoost completed training ({model.n_estimators} iterations)")
        else:
            model.fit(X_train, y_train)
            logger.info(f"XGBoost completed training ({model.n_estimators} iterations)")
            
        self.models['xgboost'] = model
        logger.info("XGBoost training completed")
        
        return model
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> lgb.LGBMClassifier:
        """
        Train LightGBM model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained LightGBM model
        """
        logger.info("Training LightGBM model...")
        
        model = self.create_lightgbm()
        
        if X_val is not None and y_val is not None:
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            
            # Check if best_iteration exists
            try:
                best_iter = model.best_iteration_
                logger.info(f"LightGBM best iteration: {best_iter}")
            except AttributeError:
                logger.info(f"LightGBM completed training ({model.n_estimators} iterations)")
        else:
            model.fit(X_train, y_train)
            logger.info(f"LightGBM completed training ({model.n_estimators} iterations)")
            
        self.models['lightgbm'] = model
        logger.info("LightGBM training completed")
        
        return model
    
    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> cb.CatBoostClassifier:
        """
        Train CatBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained CatBoost model
        """
        logger.info("Training CatBoost model...")
        
        model = self.create_catboost()
        
        if X_val is not None and y_val is not None:
            eval_set = cb.Pool(X_val, y_val)
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Check if best_iteration exists
            try:
                best_iter = model.best_iteration_
                logger.info(f"CatBoost best iteration: {best_iter}")
            except AttributeError:
                logger.info(f"CatBoost completed training ({model.get_params()['iterations']} iterations)")
        else:
            model.fit(X_train, y_train, verbose=False)
            logger.info(f"CatBoost completed training ({model.get_params()['iterations']} iterations)")
            
        self.models['catboost'] = model
        logger.info("CatBoost training completed")
        
        return model
    
    def train_all_tree_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> dict:
        """
        Train all tree-based models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training all tree-based models...")
        
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_catboost(X_train, y_train, X_val, y_val)
        
        logger.info("All tree models training completed")
        
        return self.models
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            model_name: Name of the model
            X: Features
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of the model
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        return self.models[model_name].predict_proba(X)
    
    def get_feature_importance(
        self, 
        model_name: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")
            
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f'f{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model.
        
        Args:
            model_name: Name of the model
            filepath: Path to save model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load trained model.
        
        Args:
            model_name: Name of the model
            filepath: Path to load model from
        """
        self.models[model_name] = joblib.load(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")