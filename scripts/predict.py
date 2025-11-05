"""Make predictions using trained bankruptcy prediction model."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from loguru import logger

from src.config import Config
from src.utils.helpers import set_seed, load_pickle, timer


@timer
def load_model_and_preprocessing(config, model_name):
    """Load trained model and preprocessing objects."""
    logger.info(f"Loading {model_name} model and preprocessing objects...")
    
    models_path = Path(config.get('output.models_path'))
    
    # Load preprocessing
    preprocessing_objects = load_pickle(str(models_path / 'preprocessing.pkl'))
    selected_features = load_pickle(str(models_path / 'selected_features.pkl'))
    
    # Load model
    model_path = models_path / f'{model_name}_model.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_pickle(str(model_path))
    
    logger.info("Model and preprocessing loaded successfully")
    
    return model, preprocessing_objects, selected_features


@timer
def preprocess_input_data(df, preprocessing_objects, selected_features):
    """Preprocess input data for prediction."""
    logger.info("Preprocessing input data...")
    
    scaler = preprocessing_objects['scaler']
    imputer = preprocessing_objects['imputer']
    
    # Select available features
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = set(selected_features) - set(available_features)
    
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features, filling with zeros")
        for feat in missing_features:
            df[feat] = 0
    
    X = df[selected_features]
    
    # Impute
    if imputer:
        X = pd.DataFrame(
            imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    # Scale
    if scaler:
        X = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    logger.info(f"Preprocessed data shape: {X.shape}")
    
    return X


@timer
def make_predictions(model, X, threshold=0.5):
    """Make predictions."""
    logger.info("Making predictions...")
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X)
    
    # Handle different probability formats
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
        proba_bankruptcy = y_pred_proba[:, 1]
    else:
        proba_bankruptcy = y_pred_proba
    
    # Binary predictions
    y_pred = (proba_bankruptcy >= threshold).astype(int)
    
    logger.info(f"Predictions completed: {y_pred.sum()} bankruptcies predicted out of {len(y_pred)} samples")
    
    return y_pred, proba_bankruptcy


@timer
def save_predictions(predictions, probabilities, ids, output_path):
    """Save predictions to file."""
    logger.info(f"Saving predictions to {output_path}...")
    
    results_df = pd.DataFrame({
        'id': ids,
        'predicted_bankruptcy': predictions,
        'bankruptcy_probability': probabilities,
        'risk_category': pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })
    
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved successfully")
    
    # Print summary
    logger.info("\nPrediction Summary:")
    logger.info(f"Total samples: {len(predictions)}")
    logger.info(f"Predicted bankruptcies: {predictions.sum()}")
    logger.info(f"Predicted non-bankruptcies: {len(predictions) - predictions.sum()}")
    logger.info("\nRisk Distribution:")
    logger.info(results_df['risk_category'].value_counts().to_string())


def main():
    """Main prediction script."""
    parser = argparse.ArgumentParser(description='Make bankruptcy predictions')
    parser.add_argument('--model', type=str, default='xgboost', help='Model name to use')
    parser.add_argument('--input', type=str, required=True, help='Input data file (CSV or Parquet)')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output predictions file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("BANKRUPTCY PREDICTION")
    logger.info("="*50)
    
    # Load configuration
    config = Config()
    set_seed(config.random_seed)
    
    # Load model and preprocessing
    model, preprocessing_objects, selected_features = load_model_and_preprocessing(config, args.model)
    
    # Load input data
    logger.info(f"Loading input data from {args.input}...")
    
    if args.input.endswith('.csv'):
        df_input = pd.read_csv(args.input)
    elif args.input.endswith('.parquet'):
        df_input = pd.read_parquet(args.input)
    else:
        raise ValueError("Input file must be CSV or Parquet")
    
    logger.info(f"Input data shape: {df_input.shape}")
    
    # Create IDs if not present
    if 'id' not in df_input.columns:
        ids = np.arange(len(df_input))
    else:
        ids = df_input['id'].values
        df_input = df_input.drop(columns=['id'])
    
    # Remove target if present
    if 'class' in df_input.columns:
        df_input = df_input.drop(columns=['class'])
    
    # Preprocess data
    X = preprocess_input_data(df_input, preprocessing_objects, selected_features)
    
    # Make predictions
    predictions, probabilities = make_predictions(model, X, threshold=args.threshold)
    
    # Save predictions
    save_predictions(predictions, probabilities, ids, args.output)
    
    logger.info("="*50)
    logger.info("PREDICTION COMPLETED!")
    logger.info("="*50)


if __name__ == "__main__":
    main()