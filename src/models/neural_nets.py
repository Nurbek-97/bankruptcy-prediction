"""Neural network models for bankruptcy prediction."""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib


class NeuralNetworkModel:
    """Neural network implementation for tabular data."""
    
    def __init__(self, config):
        """
        Initialize neural network.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        nn_config = self.config.get('models.neural_net', {})
        hidden_layers = nn_config.get('hidden_layers', [128, 64, 32])
        dropout_rate = nn_config.get('dropout_rate', 0.3)
        learning_rate = nn_config.get('learning_rate', 0.001)
        
        logger.info(f"Building neural network with architecture: {hidden_layers}")
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.BatchNormalization())
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.001),
                name=f'dense_{i}'
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
            
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        logger.info(f"Model built with {model.count_params()} parameters")
        
        return model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> keras.Model:
        """
        Train neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model
        """
        logger.info("Training neural network...")
        
        nn_config = self.config.get('models.neural_net', {})
        batch_size = nn_config.get('batch_size', 256)
        epochs = nn_config.get('epochs', 100)
        patience = nn_config.get('early_stopping_patience', 10)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
            
        # Calculate class weights
        class_weight = self._calculate_class_weights(y_train)
        
        # Build model
        self.model = self.build_model(X_train.shape[1])
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_auc' if validation_data else 'auc',
                patience=patience,
                restore_best_weights=True,
                mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train_scaled,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=0
        )
        
        logger.info("Neural network training completed")
        
        return self.model
    
    def _calculate_class_weights(self, y: pd.Series) -> dict:
        """
        Calculate class weights for imbalanced data.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        class_weight = dict(zip(classes, weights))
        logger.info(f"Class weights: {class_weight}")
        
        return class_weight
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled, verbose=0)
        
        return (proba >= threshold).astype(int).flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled, verbose=0).flatten()
        
        # Return in sklearn format [prob_class_0, prob_class_1]
        return np.vstack([1 - proba, proba]).T
    
    def get_training_history(self) -> dict:
        """
        Get training history.
        
        Returns:
            Training history dictionary
        """
        if self.history is None:
            raise ValueError("Model not trained")
            
        return self.history.history
    
    def save_model(self, model_path: str, scaler_path: str) -> None:
        """
        Save model and scaler.
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """
        Load model and scaler.
        
        Args:
            model_path: Path to load model from
            scaler_path: Path to load scaler from
        """
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")