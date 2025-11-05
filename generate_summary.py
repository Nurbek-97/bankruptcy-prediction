"""Generate project summary report."""

import pandas as pd
from pathlib import Path

print("="*70)
print(" "*15 + "BANKRUPTCY PREDICTION PROJECT SUMMARY")
print("="*70)

# 1. Model Comparison
print("\n1. MODEL PERFORMANCE COMPARISON:")
print("-"*70)
comparison = pd.read_csv('reports/results/model_comparison.csv', index_col=0)
print(comparison.to_string())

# 2. Best Model
best_model = comparison['roc_auc'].idxmax()
best_auc = comparison.loc[best_model, 'roc_auc']
print(f"\n✓ Best Model: {best_model.upper()} (ROC-AUC: {best_auc:.4f})")

# 3. Files Created
print("\n2. OUTPUT FILES:")
print("-"*70)

models = list(Path('models').glob('*.pkl'))
figures = list(Path('reports/figures').glob('*.png'))
results = list(Path('reports/results').glob('*.*'))

print(f"✓ Trained Models: {len(models)} files")
for m in models:
    print(f"  - {m.name}")

print(f"\n✓ Visualizations: {len(figures)} figures")
for f in figures[:5]:  # Show first 5
    print(f"  - {f.name}")
if len(figures) > 5:
    print(f"  ... and {len(figures)-5} more")

print(f"\n✓ Reports: {len(results)} files")
for r in results:
    print(f"  - {r.name}")

# 4. Next Steps
print("\n3. NEXT STEPS:")
print("-"*70)
print("✓ Review visualizations in reports/figures/")
print("✓ Check model comparison in reports/results/")
print("✓ Use scripts/predict.py for new predictions")
print("✓ Fine-tune hyperparameters in config/config.yaml")
print("✓ Deploy best model to production")

print("\n" + "="*70)
print("Project completed successfully! 🎉")
print("="*70)   