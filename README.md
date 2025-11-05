# 🏦 Bankruptcy Prediction - Advanced ML & Econometric Analysis

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

Advanced bankruptcy prediction system combining machine learning and econometric approaches for financial distress analysis.

## 🎯 Key Features

- **ML Models**: XGBoost, LightGBM, CatBoost, Neural Networks
- **Econometric Analysis**: Logistic Regression, Survival Analysis, Panel Data
- **Feature Engineering**: Financial ratios, Altman Z-Score, macro indicators
- **Ensemble Methods**: Stacking, blending, weighted averaging
- **Imbalanced Learning**: SMOTE, class weights, ensemble sampling

## 📁 Project Structure
```
bankruptcy-prediction/
├── config/              # Configuration files
├── data/                # Data storage (gitignored)
├── src/                 # Source code
│   ├── data/           # Data loading & preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Model implementations
│   ├── evaluation/     # Metrics & validation
│   └── utils/          # Utility functions
├── scripts/            # Executable scripts
├── models/             # Saved models
├── reports/            # Results & visualizations
└── tests/              # Unit tests
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### 2. Download Data
```bash
python scripts/download_data.py
```

### 3. Train Models
```bash
python scripts/train.py --config config/config.yaml
```

### 4. Evaluate
```bash
python scripts/evaluate.py --model_path models/best_model.pkl
```

### 5. Run Full Pipeline
```bash
python scripts/run_pipeline.py
```

## 📈 Model Performance

| Model | AUC-ROC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| XGBoost | - | - | - | - |
| LightGBM | - | - | - | - |
| CatBoost | - | - | - | - |
| Ensemble | - | - | - | - |

## 🔬 Methodology

### Data Processing
1. Missing value imputation
2. Outlier detection & handling
3. Feature scaling & normalization
4. Temporal validation split

### Feature Engineering
- Financial ratios (liquidity, profitability, leverage)
- Altman Z-Score & variants
- Trend features (YoY, QoQ changes)
- Industry-adjusted metrics
- Macro-economic indicators

### Model Training
- Stratified K-Fold cross-validation
- Hyperparameter optimization (Optuna)
- Class imbalance handling
- Early stopping with validation set

## 📊 Dataset

**Source**: Polish Companies Bankruptcy Dataset (UCI)
- **Samples**: 10,503 companies
- **Features**: 64 financial ratios
- **Time Period**: 5 years
- **Classes**: Bankruptcy (3.2%), Non-bankruptcy (96.8%)

## 🛠️ Technologies

- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **DL**: TensorFlow, Keras
- **Stats**: statsmodels, linearmodels
- **Feature**: feature-engine, SHAP
- **Optimization**: Optuna
- **Visualization**: matplotlib, seaborn, plotly

## 📝 License

MIT License

## 👤 Author

Nurbek Xalimjonov

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

nurbekkhalimjonov070797@gmail.com