# Depression Risk Prediction in Patients with Liver Cancer

Research-use Streamlit applications powered by trained XGBoost and calibrated
SVM models. Each interface accepts its model-aligned features, returns the
estimated probability of depression, and displays an individual SHAP force
plot.

## Deployed apps

- [XGBoost prediction app](https://liver-cancer-depression-risk-model.streamlit.app/)
- [SVM prediction app](https://older-adult-liver-cancer-depression-svm.streamlit.app/)

## XGBoost app

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/streamlit run code/app.py
```

## SVM app

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r svm/requirements.txt
.venv/bin/streamlit run svm/app.py
```

The deployment repository intentionally excludes clinical source data and
model-development scripts.

## Notice

This application is for research demonstration and model validation only. It
does not replace clinical diagnosis or treatment decisions.
