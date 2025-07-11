# AutoML CLI Tool

A Python CLI-based AutoML tool for quick EDA, preprocessing, model training, evaluation, and model selection on CSV datasets.

## Features
- Upload a CSV dataset via CLI
- Automatic EDA (shape, missing values, data types, correlation, target distribution)
- Preprocessing: missing value imputation, one-hot encoding, scaling
- Train/test split
- Model training: Logistic/Linear Regression, RandomForest, XGBoost, KNN, SVM, (optional: LightGBM)
- Model evaluation: classification and regression metrics
- Model comparison table
- Save selected model as .pkl

## Installation

1. Clone this repo or copy the files to a directory.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --csv path/to/data.csv --target target_column_name [--task classification|regression] [--test-size 0.2] [--output best_model.pkl]
```

- `--csv`: Path to your CSV file (required)
- `--target`: Name of the target column (required)
- `--task`: Task type (`classification` or `regression`). If not provided, auto-inferred.
- `--test-size`: Fraction for test set (default: 0.2)
- `--output`: Output path for saved model (default: best_model.pkl)

## Example

```bash
python main.py --csv data/iris.csv --target species --task classification
```

## Notes
- LightGBM is optional. If not installed, it will be skipped.
- The tool prints EDA and model comparison in the CLI.
- The selected model is saved as a .pkl file using joblib.

## Requirements
- Python 3.7+
- See `requirements.txt` for dependencies 