"""Test predictions on sample data."""

import pandas as pd
import numpy as np
from src.config import Config
from src.data.data_loader import BankruptcyDataLoader

# Load configuration
config = Config()

# Load some real data as example
data_loader = BankruptcyDataLoader(config)
df = data_loader.load_processed_data('raw_combined.parquet')

# Take 10 random samples
sample_data = df.sample(10, random_state=42)
sample_data.to_csv('data/sample_for_prediction.csv', index=False)

print("✓ Created sample_for_prediction.csv with 10 companies")
print("\nNow run predictions:")
print("python scripts/predict.py --model xgboost --input data/sample_for_prediction.csv --output predictions.csv")