# Depression Risk Prediction in Patients with Liver Cancer

Research-use Streamlit application powered by a trained XGBoost model. The
interface accepts 10 model-aligned features, returns the estimated probability
of depression, and displays a SHAP force plot for the individual prediction.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/streamlit run code/app.py
```

The deployment repository intentionally excludes clinical source data and
model-development scripts.

## Notice

This application is for research demonstration and model validation only. It
does not replace clinical diagnosis or treatment decisions.
