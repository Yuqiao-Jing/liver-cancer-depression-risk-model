"""Depression risk prediction for patients with liver cancer."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.json"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
BACKGROUND_PATH = BASE_DIR / "assets" / "clinical-background.jpg"

FEATURE_LABELS = {
    "Life_satisfaction": "Life Satisfaction",
    "Self_perceived_health_status": "Self-perceived Health Status",
    "Pain": "Pain",
    "Arthritis": "Arthritis",
    "Falldown": "History of Falls",
    "Gender": "Gender",
    "Kidney_disease": "Kidney Disease",
    "ADL_score": "Activities of Daily Living Score",
    "Sleep_time": "Sleep Duration",
    "Neutrophil_to_Lymphocyte_Ratio": "Neutrophil-to-Lymphocyte Ratio",
}

FEATURE_HELP = {
    "Life_satisfaction": "Enter code 1–3 according to the original data definition.",
    "Self_perceived_health_status": "Enter code 1–3 according to the original data definition.",
    "Pain": "Binary code 0/1; use the same definition as the modeling data.",
    "Arthritis": "Binary code 0/1; use the same definition as the modeling data.",
    "Falldown": "Binary code 0/1; use the same definition as the modeling data.",
    "Gender": "Binary code 0/1; the original material does not define the coding labels.",
    "Kidney_disease": "Binary code 0/1; use the same definition as the modeling data.",
    "ADL_score": "Available values: 0, 1.5, 2.5, and 5.",
    "Sleep_time": "Unit: hours; the training-data range is 4–8.",
    "Neutrophil_to_Lymphocyte_Ratio": "The training-data range is 1.4–3.5.",
}

FORCE_PLOT_LABELS = {
    "Life_satisfaction": "Life satisfaction",
    "Self_perceived_health_status": "Self-perceived health",
    "Pain": "Pain",
    "Arthritis": "Arthritis",
    "Falldown": "History of falls",
    "Gender": "Gender",
    "Kidney_disease": "Kidney disease",
    "ADL_score": "Daily living score",
    "Sleep_time": "Sleep duration",
    "Neutrophil_to_Lymphocyte_Ratio": "Neutrophil/lymphocyte ratio",
}


st.set_page_config(
    page_title="Depression Risk Prediction in Patients with Liver Cancer",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_artifacts() -> tuple[XGBClassifier, dict]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model files are missing. Run: python code/train_xgboost.py")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def image_as_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def inject_styles() -> None:
    background = image_as_data_url(BACKGROUND_PATH)
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

        :root {{
            --ink: #17364a;
            --muted: #5e7280;
            --accent: #d9666d;
            --accent-dark: #ba4f58;
            --teal: #2f7f85;
            --paper: rgba(255, 255, 255, 0.91);
        }}

        .stApp {{
            background:
                linear-gradient(180deg, rgba(237,245,244,.20) 0%, rgba(237,245,244,.32) 48%, rgba(237,245,244,.62) 100%),
                url("{background}") center top / cover fixed;
        }}

        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{ display: none; }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }}

        html, body, [class*="st-"] {{
            font-family: "Source Sans 3", "PingFang SC", sans-serif;
            color: var(--ink);
        }}

        .hero-block {{
            min-height: 430px;
            display: flex;
            justify-content: flex-start;
            align-items: center;
            flex-direction: column;
            text-align: center;
        }}

        .hero-panel {{
            width: min(1050px, 94%);
            background: rgba(255,255,255,.82);
            border: 1px solid rgba(255,255,255,.92);
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(25,60,72,.14);
            backdrop-filter: blur(9px);
            -webkit-backdrop-filter: blur(9px);
            padding: .65rem 1.1rem .75rem;
        }}

        .hero-kicker {{
            color: var(--teal);
            font-size: .79rem;
            font-weight: 700;
            letter-spacing: .12em;
            margin-bottom: .5rem;
        }}

        .hero-title {{
            color: var(--ink);
            font-family: "Noto Serif SC", serif;
            font-size: clamp(1.1rem, 1.5vw, 1.25rem) !important;
            font-weight: 700;
            line-height: 1.13 !important;
            max-width: none;
            margin: 0;
        }}

        .glass-card {{
            background: var(--paper);
            border: 1px solid rgba(255,255,255,.86);
            border-radius: 24px;
            box-shadow: 0 22px 60px rgba(25, 60, 72, .18);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            padding: 1.35rem 1.55rem .3rem;
            margin-bottom: .9rem;
        }}

        .section-title {{
            font-family: "Noto Serif SC", serif;
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: .2rem;
        }}

        [data-testid="stForm"] {{
            background: rgba(255,255,255,.91);
            border: 1px solid rgba(255,255,255,.88);
            border-radius: 24px;
            box-shadow: 0 22px 60px rgba(25, 60, 72, .18);
            backdrop-filter: blur(15px);
            padding: 1.35rem 1.55rem 1.25rem;
        }}

        div[data-testid="stSelectbox"] label,
        div[data-testid="stNumberInput"] label {{
            color: #203c4d !important;
            font-family: "Noto Serif SC", serif;
            font-weight: 600 !important;
            font-size: .96rem !important;
        }}

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {{
            background: #f1f5f7 !important;
            border-color: #dfe8eb !important;
        }}

        div[data-baseweb="select"] > div {{
            min-height: 44px;
            border-radius: 10px;
        }}

        div[data-testid="stNumberInput"] > div > div {{
            border-radius: 10px;
            overflow: hidden;
        }}

        div[data-testid="stFormSubmitButton"] button {{
            background: linear-gradient(135deg, var(--accent), #e27b76);
            border: 1px solid rgba(255,255,255,.6);
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(186, 79, 88, .30);
            color: white;
            font-size: 1.02rem;
            font-weight: 700;
            min-height: 48px;
            transition: transform .18s ease, box-shadow .18s ease;
        }}

        div[data-testid="stFormSubmitButton"] button:hover {{
            background: linear-gradient(135deg, var(--accent-dark), var(--accent));
            border-color: rgba(255,255,255,.8);
            box-shadow: 0 10px 28px rgba(186, 79, 88, .38);
            transform: translateY(-1px);
        }}

        .result-card {{
            background: rgba(255,255,255,.98);
            border: 1px solid rgba(255,255,255,.9);
            border-left: 6px solid var(--accent);
            border-radius: 18px;
            box-shadow: 0 16px 45px rgba(25,60,72,.18);
            padding: 1.25rem 1.35rem;
            margin-top: 1rem;
        }}

        .result-eyebrow {{
            color: var(--teal);
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .13em;
            text-transform: uppercase;
        }}

        .result-copy {{
            color: var(--ink);
            font-family: "Noto Serif SC", serif;
            font-size: clamp(1.3rem, 2.4vw, 2rem);
            line-height: 1.45;
            margin-top: .35rem;
        }}

        .probability {{ color: var(--accent-dark); font-weight: 700; }}

        .st-key-result_visuals {{
            background: rgba(255,255,255,.98);
            border: 1px solid rgba(255,255,255,.94);
            border-radius: 18px;
            box-shadow: 0 16px 45px rgba(25,60,72,.18);
            padding: 1rem 1.2rem .7rem;
            margin-top: 1rem;
        }}

        .st-key-result_visuals [data-testid="stImage"] img {{
            background: #ffffff;
            border-radius: 12px;
        }}

        @media (max-width: 760px) {{
            [data-testid="stMainBlockContainer"] {{ padding: 1.25rem .8rem 2rem; }}
            .hero-block {{ min-height: 290px; }}
            .hero-panel {{ padding: .85rem 1rem 1rem; }}
            .hero-title {{ font-size: 1.55rem !important; }}
            [data-testid="stForm"] {{ padding: 1rem; border-radius: 18px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def coded_select(feature: str, values: list[float], default: float) -> float:
    normalized = [int(value) if float(value).is_integer() else value for value in values]
    normalized_default = int(default) if float(default).is_integer() else default
    default_index = normalized.index(normalized_default)
    selected = st.selectbox(
        FEATURE_LABELS[feature],
        options=normalized,
        index=default_index,
        help=FEATURE_HELP[feature],
        format_func=str,
    )
    return float(selected)


def force_plot_html(model: XGBClassifier, row: pd.DataFrame) -> str:
    """Create a responsive force plot for one observation."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    expected_value = float(explainer.expected_value)
    display_values = row.iloc[0].copy()
    display_values.index = [FORCE_PLOT_LABELS[name] for name in row.columns]

    visualizer = shap.force_plot(
        expected_value,
        shap_values[0],
        display_values,
        feature_names=list(display_values.index),
        out_names="Predicted probability",
        link="logit",
        matplotlib=False,
        contribution_threshold=0.10,
    )
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        {shap.getjs()}
        <style>
          html, body {{ margin: 0; padding: 0; background: #ffffff; overflow: hidden; }}
          body {{ font-family: Arial, sans-serif; color: #17364a; }}
          .force-title {{
            font-size: 15px;
            font-weight: 700;
            margin: 2px 0 18px 4px;
          }}
          .force-wrap {{ width: 100%; min-width: 0; overflow: hidden; padding-top: 4px; }}
          .force-wrap svg .tick text {{ display: none !important; }}
          .force-wrap svg .tick line {{ opacity: 0.18 !important; }}
        </style>
      </head>
      <body>
        <div class="force-title">Feature Impact Force Plot</div>
        <div class="force-wrap">{visualizer.html(label_margin=28)}</div>
      </body>
    </html>
    """


def gauge_figure(probability: float, threshold: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={"x": [0.08, 0.92], "y": [0.02, 0.92]},
            number={
                "suffix": "%",
                "valueformat": ".1f",
                "font": {"color": "#17364a", "size": 30},
            },
            title={"text": "Predicted Probability", "font": {"color": "#5e7280", "size": 15}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 50, 100],
                    "ticktext": ["0%", "50%", "100%"],
                    "tickfont": {"size": 10, "color": "#667985"},
                },
                "bar": {"color": "#d9666d"},
                "bgcolor": "rgba(255,255,255,.35)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(47,127,133,.16)"},
                    {"range": [threshold * 100, 100], "color": "rgba(217,102,109,.13)"},
                ],
                "threshold": {
                    "line": {"color": "#17364a", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(
        height=270,
        margin=dict(l=48, r=48, t=54, b=18),
        paper_bgcolor="#ffffff",
        font=dict(family="Source Sans 3, PingFang SC"),
    )
    return figure


def main() -> None:
    inject_styles()
    try:
        model, metadata = load_artifacts()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.error(html.escape(str(exc)))
        st.stop()

    st.markdown(
        """
        <div class="hero-block">
            <div class="hero-panel">
                <div class="hero-kicker">Mental Health Assessment for Patients with Liver Cancer</div>
                <h1 class="hero-title">Depression Risk Prediction in Patients with Liver Cancer</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    specs = metadata["feature_specs"]
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown('<div class="section-title">Patient Features</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        values: dict[str, float] = {}
        left_features = metadata["features"][:5]
        right_features = metadata["features"][5:]

        with left:
            for feature in left_features:
                spec = specs[feature]
                values[feature] = coded_select(feature, spec["values"], spec["default"])

        with right:
            for feature in right_features:
                spec = specs[feature]
                if feature == "Neutrophil_to_Lymphocyte_Ratio":
                    values[feature] = st.number_input(
                        FEATURE_LABELS[feature],
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(spec["default"]),
                        step=0.1,
                        format="%.1f",
                        help=FEATURE_HELP[feature],
                    )
                else:
                    values[feature] = coded_select(feature, spec["values"], spec["default"])

        submitted = st.form_submit_button("Predict", width="stretch")

    if submitted:
        ordered_row = pd.DataFrame(
            [[values[feature] for feature in metadata["features"]]],
            columns=metadata["features"],
        )
        probability = float(model.predict_proba(ordered_row)[0, 1])
        threshold = float(metadata["decision_threshold"])
        classification = "Higher Risk" if probability >= threshold else "Lower Risk"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-eyebrow">Prediction Result</div>
                <div class="result-copy">Based on the current features, the predicted probability of depression is
                    <span class="probability">{probability:.1%}</span>, classified as
                    <span class="probability">{classification}</span>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="result_visuals"):
            gauge_spacer_left, gauge_column, gauge_spacer_right = st.columns([1, 1.15, 1])
            with gauge_column:
                st.plotly_chart(
                    gauge_figure(probability, threshold),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            st.iframe(
                force_plot_html(model, ordered_row),
                width="stretch",
                height=250,
            )


if __name__ == "__main__":
    main()
