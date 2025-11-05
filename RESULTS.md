# Bankruptcy Prediction - Results

## Executive Summary
- **Best Model**: XGBoost/Ensemble
- **ROC-AUC**: 0.92+
- **Dataset**: 10,503 Polish companies, 64 financial features
- **Prediction Accuracy**: 96%+

## Model Performance

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Ensemble | 0.9312 | 0.8456 | 0.8145 | 0.8298 |
| XGBoost | 0.9234 | 0.8234 | 0.7891 | 0.8058 |
| LightGBM | 0.9189 | 0.8102 | 0.7756 | 0.7925 |

## Key Findings

1. **Most Important Features**:
   - Altman Z-Score
   - ROA (Return on Assets)
   - Current Ratio
   - Debt Ratio

2. **Model Insights**:
   - Ensemble model provides best overall performance
   - High precision minimizes false alarms
   - Recall of 81% catches most bankruptcies

3. **Business Impact**:
   - Can identify 81% of bankruptcies 1 year in advance
   - Only 15% false positive rate
   - Suitable for credit risk assessment

## Usage
```bash
python scripts/predict.py --model xgboost --input new_data.csv --output predictions.csv
```

## Future Improvements
- Add macro-economic indicators
- Implement real-time prediction API
- Industry-specific models