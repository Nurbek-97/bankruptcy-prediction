"""Train bankruptcy prediction models."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.config import Config
from src.data.data_loader import BankruptcyDataLoader
from src.data.preprocessing import DataPreprocessor
from src.data.feature_selection import FeatureSelector
from src.features.feature_engineering import FeatureEngineer
from src.models.baseline import BaselineModels
from src.models.tree_models import TreeModels
from src.models.neural_nets import NeuralNetworkModel
from src.models.ensemble import EnsembleModel
from src.evaluation.metrics import ModelMetrics
from src.utils.helpers import set_seed, save_pickle, timer


@timer
def load_and_preprocess_data(config):
    """Load and preprocess data."""
    logger.info("Loading and preprocessing data...")
    
    # Load data
    data_loader = BankruptcyDataLoader(config)
    df = data_loader.load_processed_data('raw_combined.parquet')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Handle outliers
    df_clean = preprocessor.handle_outliers(df_clean)
    
    # Impute missing values
    df_clean = preprocessor.impute_missing_values(df_clean)
    
    logger.info(f"Preprocessed data shape: {df_clean.shape}")
    
    return df_clean, preprocessor


@timer
def engineer_features(df, config):
    """Engineer features."""
    logger.info("Engineering features...")
    
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer_features(df)
    
    # Create aggregated features
    df_engineered = feature_engineer.create_aggregated_features(df_engineered)
    
    logger.info(f"Engineered data shape: {df_engineered.shape}")
    
    return df_engineered


@timer
def split_and_scale_data(df, config, preprocessor):
    """Split and scale data."""
    logger.info("Splitting and scaling data...")
    
    # Temporal split
    splits = preprocessor.create_temporal_split(df)
    
    train_df = splits['train']
    val_df = splits['val']
    test_df = splits['test']
    
    # Split features and target
    X_train, y_train = preprocessor.split_features_target(train_df)
    X_val, y_val = preprocessor.split_features_target(val_df)
    X_test, y_test = preprocessor.split_features_target(test_df)
    
    # Feature selection
    feature_selector = FeatureSelector(config)
    X_train = feature_selector.remove_low_variance(X_train)
    X_train = feature_selector.remove_correlated_features(X_train)
    
    # Apply same feature selection to val and test
    selected_features = X_train.columns.tolist()
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]
    
    # CRITICAL: Remove any remaining NaN values created by feature engineering
    logger.info("Checking for NaN values...")
    nan_counts_before = X_train.isnull().sum().sum()
    
    if nan_counts_before > 0:
        logger.warning(f"Found {nan_counts_before} NaN values, filling with median...")
        
        # Fill NaN with median for each column
        for col in X_train.columns:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_val[col] = X_val[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    # Check for infinite values
    logger.info("Checking for infinite values...")
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Fill any inf-converted NaN
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    # Final verification
    nan_counts_after = X_train.isnull().sum().sum()
    inf_counts = np.isinf(X_train.values).sum()
    
    logger.info(f"NaN values after cleaning: {nan_counts_after}")
    logger.info(f"Inf values after cleaning: {inf_counts}")
    
    if nan_counts_after > 0 or inf_counts > 0:
        logger.error("Still have NaN/Inf values! Dropping problematic rows...")
        # Last resort: drop rows with NaN
        valid_idx = X_train.notna().all(axis=1)
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
    
    # Scale features
    X_train_scaled = preprocessor.scale_features(X_train, fit=True)
    X_val_scaled = preprocessor.scale_features(X_val, fit=False)
    X_test_scaled = preprocessor.scale_features(X_test, fit=False)
    
    logger.info(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    
    return (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test), selected_features


@timer
def handle_imbalance(X_train, y_train, config):
    """Handle class imbalance."""
    if not config.get('models.handle_imbalance', True):
        return X_train, y_train
        
    logger.info("Handling class imbalance...")
    
    strategy = config.get('models.sampling_strategy', 'smote')
    
    logger.info(f"Class distribution before: {y_train.value_counts().to_dict()}")
    
    if strategy == 'smote':
        sampler = SMOTE(random_state=config.random_seed)
    elif strategy == 'smote_tomek':
        sampler = SMOTETomek(random_state=config.random_seed)
    elif strategy == 'random_under':
        sampler = RandomUnderSampler(random_state=config.random_seed)
    else:
        logger.warning(f"Unknown sampling strategy: {strategy}, skipping")
        return X_train, y_train
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    logger.info(f"Class distribution after: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    return X_resampled, y_resampled


@timer
def train_baseline_models(X_train, y_train, X_val, y_val, config):
    """Train baseline models."""
    logger.info("Training baseline models...")
    
    baseline = BaselineModels(config)
    models = baseline.train_all_baselines(X_train, y_train, X_val, y_val)
    
    return models


@timer
def train_tree_models(X_train, y_train, X_val, y_val, config):
    """Train tree-based models."""
    logger.info("Training tree-based models...")
    
    tree_models = TreeModels(config)
    models = tree_models.train_all_tree_models(X_train, y_train, X_val, y_val)
    
    return models, tree_models


@timer
def train_neural_network(X_train, y_train, X_val, y_val, config):
    """Train neural network."""
    logger.info("Training neural network...")
    
    try:
        nn_model = NeuralNetworkModel(config)
        model = nn_model.train(X_train, y_train, X_val, y_val)
        return nn_model
    except ImportError:
        logger.warning("TensorFlow not available, skipping neural network")
        return None


@timer
def train_ensemble(base_models, X_train, y_train, X_val, y_val, config):
    """Train ensemble model."""
    logger.info("Training ensemble model...")
    
    ensemble = EnsembleModel(config)
    
    ensemble_method = config.get('ensemble.method', 'stacking')
    
    if ensemble_method == 'stacking':
        # Create fresh models without early stopping for stacking
        logger.info("Creating base models for stacking (without early stopping)...")
        
        stacking_base_models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=30,
                random_state=config.random_seed,
                n_jobs=-1,
                tree_method='hist'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                is_unbalance=True,
                random_state=config.random_seed,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.01,
                l2_leaf_reg=3,
                subsample=0.8,
                auto_class_weights='Balanced',
                random_state=config.random_seed,
                verbose=False,
                thread_count=-1
            )
        }
        
        ensemble_model = ensemble.train_stacking_ensemble(stacking_base_models, X_train, y_train)
        
    elif ensemble_method == 'voting':
        ensemble_model = ensemble.train_voting_ensemble(base_models, X_train, y_train)
    elif ensemble_method == 'weighted':
        ensemble.create_weighted_average_ensemble(base_models)
        ensemble_model = None
    else:
        logger.warning(f"Unknown ensemble method: {ensemble_method}")
        ensemble_model = None
    
    return ensemble, ensemble_model


@timer
def evaluate_all_models(models, X_val, y_val, config):
    """Evaluate all models."""
    logger.info("Evaluating all models...")
    
    metrics_calculator = ModelMetrics(config)
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        try:
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            metrics = metrics_calculator.calculate_all_metrics(y_val, y_pred, y_pred_proba)
            results[model_name] = metrics
            
            # Print metrics
            metrics_calculator.print_metrics(metrics, model_name)
            
            # Plot confusion matrix
            metrics_calculator.plot_confusion_matrix(y_val, y_pred, model_name)
            
            # Plot ROC curve
            metrics_calculator.plot_roc_curve(y_val, y_pred_proba, model_name)
            
            # Plot PR curve
            metrics_calculator.plot_precision_recall_curve(y_val, y_pred_proba, model_name)
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
    
    # Compare models
    comparison_df = metrics_calculator.compare_models(results)
    
    return results, comparison_df


@timer
def save_models(models, tree_models_obj, nn_model, ensemble_obj, preprocessor, selected_features, config):
    """Save all trained models."""
    logger.info("Saving models...")
    
    models_path = Path(config.get('output.models_path'))
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Save tree models
    for name in ['xgboost', 'lightgbm', 'catboost']:
        if name in models:
            tree_models_obj.save_model(name, str(models_path / f'{name}_model.pkl'))
    
    # Save neural network
    if nn_model and nn_model.model:
        nn_model.save_model(
            str(models_path / 'neural_net_model.h5'),
            str(models_path / 'neural_net_scaler.pkl')
        )
    
    # Save ensemble
    if ensemble_obj:
        ensemble_obj.save_ensemble(str(models_path / 'ensemble_model.pkl'))
    
    # Save preprocessor
    preprocessing_objects = preprocessor.get_preprocessing_pipeline()
    save_pickle(preprocessing_objects, str(models_path / 'preprocessing.pkl'))
    
    # Save selected features
    save_pickle(selected_features, str(models_path / 'selected_features.pkl'))
    
    logger.info("All models saved successfully!")


def main():
    """Main training pipeline."""
    logger.info("="*50)
    logger.info("BANKRUPTCY PREDICTION MODEL TRAINING")
    logger.info("="*50)
    
    # Load configuration
    config = Config()
    set_seed(config.random_seed)
    
    # Load and preprocess data
    df, preprocessor = load_and_preprocess_data(config)
    
    # Engineer features
    df_engineered = engineer_features(df, config)
    
    # Split and scale data
    (X_train, y_train, X_val, y_val, X_test, y_test), selected_features = split_and_scale_data(
        df_engineered, config, preprocessor
    )
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train, config)
    
    # Train baseline models
    baseline_models = train_baseline_models(X_train_balanced, y_train_balanced, X_val, y_val, config)
    
    # Train tree models
    tree_models_dict, tree_models_obj = train_tree_models(X_train_balanced, y_train_balanced, X_val, y_val, config)
    
    # Train neural network
    nn_model = train_neural_network(X_train_balanced, y_train_balanced, X_val, y_val, config)
    
    # Combine all models
    all_models = {**baseline_models, **tree_models_dict}
    if nn_model and nn_model.model:
        all_models['neural_net'] = nn_model
    
    # Train ensemble
    if config.get('ensemble.enabled', True):
        ensemble_obj, ensemble_model = train_ensemble(tree_models_dict, X_train_balanced, y_train_balanced, X_val, y_val, config)
        if ensemble_model:
            all_models['ensemble'] = ensemble_obj
    else:
        ensemble_obj = None
    
    # Evaluate all models
    results, comparison_df = evaluate_all_models(all_models, X_val, y_val, config)
    
    # Save models
    save_models(all_models, tree_models_obj, nn_model, ensemble_obj, preprocessor, selected_features, config)
    
    # Final evaluation on test set
    logger.info("="*50)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*50)
    
    best_model_name = comparison_df.index[0]
    best_model = all_models[best_model_name]
    
    logger.info(f"Best model: {best_model_name}")
    
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    
    metrics_calculator = ModelMetrics(config)
    test_metrics = metrics_calculator.calculate_all_metrics(y_test, y_test_pred, y_test_proba)
    metrics_calculator.print_metrics(test_metrics, f"{best_model_name} (Test Set)")
    
    logger.info("="*50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*50)


if __name__ == "__main__":
    main()