# Depression Risk Prediction in Patients with Liver Cancer

This Streamlit application provides an individual prediction from a trained
XGBoost model and displays a SHAP force plot for the current prediction. The
application uses the same 10 features, coding, and units as the modeling data.

## Run locally

From the repository root, run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/streamlit run code/app.py
```

Then visit `http://localhost:8501`.

## Deployment files

- `app.py`: Streamlit interface and prediction logic.
- `models/xgb_model.json`: trained XGBoost model.
- `models/metadata.json`: feature order, ranges, and model metadata.
- `assets/clinical-background.jpg`: application background image.

The clinical source data and model-development scripts are intentionally
excluded from the deployment repository.

## Research-use notice

This application is for research demonstration and model validation only. It
does not replace clinical diagnosis or treatment decisions.
