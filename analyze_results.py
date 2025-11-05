"""Analyze trained model results."""

import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load best model (XGBoost)
model = joblib.load('models/xgboost_model.pkl')

# Get feature importance
importance_df = pd.DataFrame({
    'feature': model.feature_names_in_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Display top 20 features
print("="*60)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*60)
print(importance_df.head(20).to_string(index=False))

# Save to CSV
importance_df.to_csv('reports/results/feature_importance_detailed.csv', index=False)
print("\n✓ Saved to reports/results/feature_importance_detailed.csv")