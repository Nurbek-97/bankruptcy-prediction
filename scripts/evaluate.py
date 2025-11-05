"""Evaluate trained bankruptcy prediction models."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from loguru import logger

from src.config import Config
from src.data.data_loader import BankruptcyDataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.evaluation.metrics import ModelMetrics
from src.utils.helpers import set_seed, load_pickle, timer


@timer
def load_test_data(config):
    """Load test data."""
    logger.info("Loading test data...")
    
    data_loader = BankruptcyDataLoader(config)
    df = data_loader.load_processed_data('raw_combined.parquet')
    
    # Use last year as test set
    test_year = df['year'].max()
    df_test = df[df['year'] == test_year]
    
    logger.info(f"Test data shape: {df_test.shape}")
    
    return df_test


@timer
def preprocess_and_engineer_features(df_test, config):
    """Apply same preprocessing and feature engineering as training."""
    logger.info("Applying preprocessing and feature engineering...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Clean data
    df_clean = preprocessor.clean_data(df_test)
    
    # Handle outliers
    df_clean = preprocessor.handle_outliers(df_clean)
    
    # Impute missing values
    df_clean = preprocessor.impute_missing_values(df_clean)
    
    # Feature engineering
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer_features(df_clean)
    df_engineered = feature_engineer.create_aggregated_features(df_engineered)
    
    # Handle NaN from feature engineering
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_engineered[col].isnull().any():
            df_engineered[col] = df_engineered[col].ffill().bfill().fillna(0)
    
    # Replace inf
    df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        median_val = df_engineered[col].median()
        df_engineered[col] = df_engineered[col].fillna(median_val)
    
    logger.info(f"Engineered data shape: {df_engineered.shape}")
    
    return df_engineered, preprocessor


@timer
def prepare_test_data(df_test, preprocessing_objects, selected_features, preprocessor):
    """Prepare test data with saved preprocessing."""
    logger.info("Preparing test data...")
    
    # Get scaler
    scaler = preprocessing_objects['scaler']
    
    # Separate features and target
    if 'class' in df_test.columns:
        y_test = df_test['class']
        X_test = df_test.drop(columns=['class'])
    else:
        y_test = None
        X_test = df_test
    
    # Drop year if present
    if 'year' in X_test.columns:
        X_test = X_test.drop(columns=['year'])
    
    # Select only the features that were selected during training
    available_features = [f for f in selected_features if f in X_test.columns]
    missing_features = set(selected_features) - set(available_features)
    
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features from training")
        # Add missing features as zeros
        for feat in missing_features:
            X_test[feat] = 0
    
    # Ensure correct order
    X_test = X_test[selected_features]
    
    # Scale
    if scaler:
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
    logger.info(f"Prepared test data shape: {X_test.shape}")
    
    return X_test, y_test


@timer
def evaluate_model(model, X_test, y_test, model_name, config):
    """Evaluate a single model."""
    logger.info(f"Evaluating {model_name}...")
    
    metrics_calculator = ModelMetrics(config)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred, y_pred_proba)
    
    # Print metrics
    metrics_calculator.print_metrics(metrics, model_name)
    
    # Save classification report
    metrics_calculator.save_classification_report(y_test, y_pred, model_name)
    
    # Plot confusion matrix
    metrics_calculator.plot_confusion_matrix(y_test, y_pred, model_name)
    
    # Plot ROC curve
    metrics_calculator.plot_roc_curve(y_test, y_pred_proba, model_name)
    
    # Plot PR curve
    metrics_calculator.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
    
    # Find optimal threshold
    optimal_threshold = metrics_calculator.find_optimal_threshold(
        y_test, 
        y_pred_proba,
        metric='f1'
    )
    
    return metrics, optimal_threshold


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate bankruptcy prediction model')
    parser.add_argument('--model', type=str, default='xgboost', help='Model name to evaluate')
    parser.add_argument('--model_path', type=str, default=None, help='Custom model path')
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("MODEL EVALUATION")
    logger.info("="*50)
    
    # Load configuration
    config = Config()
    set_seed(config.random_seed)
    
    models_path = Path(config.get('output.models_path'))
    
    # Load preprocessing objects
    preprocessing_objects = load_pickle(str(models_path / 'preprocessing.pkl'))
    selected_features = load_pickle(str(models_path / 'selected_features.pkl'))
    
    # Load test data
    df_test = load_test_data(config)
    
    # Apply same preprocessing and feature engineering
    df_engineered, preprocessor = preprocess_and_engineer_features(df_test, config)
    
    # Prepare test data
    X_test, y_test = prepare_test_data(df_engineered, preprocessing_objects, selected_features, preprocessor)
    
    if y_test is None:
        logger.error("No target variable found in test data!")
        return
    
    # Load model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = models_path / f'{args.model}_model.pkl'
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    logger.info(f"Loading model from {model_path}...")
    model = load_pickle(str(model_path))
    
    # Evaluate model
    metrics, optimal_threshold = evaluate_model(model, X_test, y_test, args.model, config)
    
    logger.info("="*50)
    logger.info("EVALUATION COMPLETED!")
    logger.info("="*50)


if __name__ == "__main__":
    main()