# SVM Depression Risk Prediction in Patients with Liver Cancer

This Streamlit application loads the frozen calibrated SVM model, accepts the
five features used during model development, returns the predicted probability
for outcome 1, and displays an individual SHAP force plot.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r streamlit_svm/requirements.txt
.venv/bin/streamlit run streamlit_svm/app.py
```

The deployment package intentionally excludes all source patient datasets.

## Notice

This application is intended for research demonstration and model validation.
It does not replace clinical diagnosis or treatment decisions.

