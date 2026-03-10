# NovaPay Fraud Detection (Streamlit)

A small Streamlit web app that loads a pre-trained fraud detection model and lets you enter transaction details to predict whether a transaction is **fraudulent**.

## How it works

- **Model loading**
  - On startup, the app loads:
    - `fraud_model.pkl` (the trained classifier)
    - `model_features.pkl` (the feature column order used during training)
  - Loading is done with `joblib.load(...)`.

- **User input**
  - The UI collects five transaction fields:
    - `amount`
    - `hour`
    - `customer_txn_count`
    - `avg_customer_amount`
    - `merchant_risk`

- **Feature alignment**
  - Inputs are assembled into a one-row `pandas.DataFrame`.
  - The DataFrame is then aligned to the exact training feature list via:
    - `input_data = input_data.reindex(columns=features, fill_value=0)`
  - This ensures the model receives the same column set and ordering as during training.

- **Prediction**
  - When you click **Predict Fraud**, the app runs:
    - `model.predict(...)` for the fraud/legit class
    - `model.predict_proba(...)` for the fraud probability
  - The app displays a status message plus a probability metric.

## Project structure

- `streamlit_app.py` - Streamlit application entry point
- `fraud_model.pkl` - Serialized trained model (loaded at runtime)
- `model_features.pkl` - Serialized list of training feature names / ordering
- `venv/` - Local virtual environment (optional; not required if you create your own)

## Requirements

### System

- Python 3.9+ (3.10/3.11 recommended)

### Python packages

At minimum, the app imports and needs:

- `streamlit`
- `pandas`
- `joblib`

In most cases, the serialized model in `fraud_model.pkl` is a scikit-learn estimator/pipeline, so you will typically also need:

- `scikit-learn`

If your model was trained with additional libraries (e.g. `xgboost`, `lightgbm`, etc.), install those too, otherwise `joblib.load` can fail.

## Setup

1. (Recommended) Create and activate a virtual environment.

2. Install dependencies:

```bash
pip install streamlit pandas joblib scikit-learn
```

3. Verify the model artifacts exist in the project root:

- `fraud_model.pkl`
- `model_features.pkl`

## Run the app

From the project directory:

```bash
streamlit run streamlit_app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## Troubleshooting

- **`FileNotFoundError` for `fraud_model.pkl` or `model_features.pkl`**
  - Run Streamlit from the repository root so relative paths resolve correctly.

- **`ModuleNotFoundError` during `joblib.load`**
  - The model was serialized with a library that isn’t installed in your environment.
  - Install the missing dependency (common one is `scikit-learn`).

- **Different features than expected**
  - `model_features.pkl` is treated as the source of truth for the model input columns.
  - If you retrain the model, regenerate and commit an updated `model_features.pkl`.
