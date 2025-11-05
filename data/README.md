# Data Directory

## Structure

- `raw/` - Raw downloaded data (gitignored)
- `processed/` - Cleaned and processed data (gitignored)
- `interim/` - Intermediate data during processing (gitignored)
- `external/` - External datasets (gitignored)

## Data Source

Polish Companies Bankruptcy Dataset from UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
- **Samples**: 10,503 companies
- **Features**: 64 financial ratios
- **Period**: 5 years (2000-2013)

## Download Data
```bash
python scripts/download_data.py
```

## Note

Data files are gitignored due to size. Run download script to fetch data.