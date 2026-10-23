import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "final_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

st.set_page_config(
    page_title="Credit Card Fraud Detection & Risk Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM UI
# ============================================================
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background:
          radial-gradient(circle at 75% 0%, rgba(12, 70, 132, .17), transparent 35%),
          #061326;
        color: #e7f0ff;
    }
    [data-testid="stHeader"] { background: rgba(5, 17, 34, .96); }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081b35 0%, #07152b 100%);
        border-right: 1px solid #17375c;
    }
    [data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1800px; }
    h1, h2, h3 { color: #f4f8ff !important; letter-spacing: -.35px; }
    p, label, .stMarkdown { color: #c4d6f2; }
    .topbar {
        position:relative; z-index:2; margin-top:0;
        display:flex; align-items:center; justify-content:space-between;
        gap:18px; padding: 5px 0 16px 0; margin-bottom: 18px;
        border-bottom: 1px solid #17375c;
    }
    .brand { display:flex; align-items:center; gap:14px; }
    .brand-icon {
        width:48px; height:48px; display:flex; align-items:center;
        justify-content:center; border-radius:14px;
        background:linear-gradient(135deg,#2c91ff,#1654d8);
        box-shadow:0 0 25px rgba(39,126,255,.22); font-size:27px;
    }
    .brand-title { font-size:24px; font-weight:800; color:#f5f9ff !important; line-height:1.2; text-shadow:0 1px 8px rgba(0,0,0,.35); }
    .brand-subtitle { font-size:12px; color:#b7cceb !important; margin-top:4px; }
    .prototype { color:#9dbbe4; font-size:12px; white-space:nowrap; }
    .welcome-title { font-size:23px; font-weight:750; color:#f4f8ff; margin-bottom:3px; }
    .muted { color:#9db5d8; font-size:13px; }
    .metric {
        min-height:118px; border-radius:13px; padding:17px 16px;
        border:1px solid #20446f; background:linear-gradient(145deg,#0d2444,#0a1a33);
    }
    .metric.blue { border-color:#1d64bb; background:linear-gradient(145deg,#0b2b59,#0b1b35); }
    .metric.red { border-color:#8b2447; background:linear-gradient(145deg,#30172e,#101b34); }
    .metric.green { border-color:#08795f; background:linear-gradient(145deg,#063b3b,#0a1b33); }
    .metric.purple { border-color:#5839a6; background:linear-gradient(145deg,#21174c,#0c1c37); }
    .metric-label { color:#d4e2f7; font-size:12px; margin-bottom:9px; }
    .metric-value { color:#f7faff; font-size:clamp(17px, 1.25vw, 25px); font-weight:800; line-height:1.12; overflow-wrap:anywhere; }
    .metric-note { color:#9db5d8; font-size:11px; margin-top:8px; }
    .panel {
        border:1px solid #1c3a60; border-radius:12px;
        background:rgba(9, 28, 53, .72); padding:14px 15px; margin-bottom:14px;
    }
    .panel-title { color:#f1f6ff; font-size:14px; font-weight:750; margin-bottom:4px; }
    .panel-subtitle { color:#8eadd7; font-size:11px; margin-bottom:12px; }
    .risk-card {
        padding:16px; border:1px solid #087e70; border-radius:12px;
        background:linear-gradient(130deg,rgba(0,91,83,.34),rgba(7,31,52,.9));
    }
    .risk-title { color:#46f0c5; font-weight:750; font-size:13px; }
    .risk-score { color:#f7fbff; font-size:29px; font-weight:850; margin:4px 0; }
    .risk-label { color:#46f0c5; font-size:12px; }
    .small-note { color:#8ea9ce; font-size:11px; line-height:1.5; }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius:8px; font-weight:700; min-height:40px;
        border:1px solid #3288ff; background:linear-gradient(90deg,#2187ff,#3475ff);
        color:white;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        border-color:#70b4ff; color:white; background:#2468db;
    }
    div[data-testid="stForm"] {
        border:1px solid #1b3d64; border-radius:10px; padding:12px;
        background:rgba(8,24,46,.55);
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background:#0b203d !important; border-color:#284b73 !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap:16px; border-bottom:1px solid #1d3c60; }
    .stTabs [data-baseweb="tab"] { color:#9dbbe4; }
    .stTabs [aria-selected="true"] { color:#45a5ff !important; }
    [data-testid="stDataFrame"] { border:1px solid #1c3a60; border-radius:9px; }
    hr { border-color:#1b3657; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL / DATA / REPORTS
# ============================================================
@st.cache_resource
def load_model_files():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, {}, "Model files missing. Run `python -m src.evaluate` first."
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return joblib.load(MODEL_PATH), meta, None
    except Exception as exc:
        return None, {}, f"Unable to load model files: {exc}"


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None


@st.cache_data
def load_csv_report(name):
    path = REPORTS / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


model, metadata, model_error = load_model_files()
data = load_dataset()
FEATURES = metadata.get(
    "feature_columns",
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
)
THRESHOLD = float(metadata.get("threshold", 0.5))
MODEL_NAME = str(metadata.get("selected_model", "Trained model"))

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def metric_card(label, value, note, style="blue", icon=""):
    st.markdown(
        f"""<div class="metric {style}">
        <div class="metric-label">{icon} &nbsp; {label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def plot_theme(fig, height=240):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=35, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4d6f2", size=10),
        legend=dict(font=dict(size=9), orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(95,135,180,.14)", zerolinecolor="#29496c")
    fig.update_yaxes(gridcolor="rgba(95,135,180,.14)", zerolinecolor="#29496c")
    return fig


def predict_values(values):
    if model is None:
        raise RuntimeError(model_error or "Model unavailable.")
    frame = pd.DataFrame([values]).reindex(columns=FEATURES)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("All model features must have numeric values.")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("This app expects a model with predict_proba().")
    score = float(model.predict_proba(frame)[0][1])
    flagged = score >= THRESHOLD
    record = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Amount": float(values.get("Amount", 0)),
        "V1": float(values.get("V1", 0)),
        "V2": float(values.get("V2", 0)),
        "V3": float(values.get("V3", 0)),
        "Risk Score": score,
        "Prediction": "Flagged for review" if flagged else "Below threshold",
    }
    st.session_state.prediction_history.insert(0, record)
    st.session_state.prediction_history = st.session_state.prediction_history[:50]
    st.session_state.last_prediction = {
        "score": score, "flagged": flagged, "amount": values.get("Amount", 0)
    }
    return score, flagged


def transaction_form(key_prefix="dash", compact=True):
    with st.form(f"{key_prefix}_prediction_form"):
        time_val = st.number_input("Time", min_value=0.0, value=12345.0,
                                   step=60.0, key=f"{key_prefix}_time")
        amount = st.number_input("Amount", min_value=0.0, value=100.0,
                                  step=10.0, key=f"{key_prefix}_amount")
        st.markdown('<div class="small-note">Model features</div>',
                    unsafe_allow_html=True)
        # Display all 28 anonymized features in a compact expander.
        with st.expander("Edit V1–V28 features", expanded=False):
            feature_values = {}
            cols = st.columns(3)
            for i in range(1, 29):
                with cols[(i - 1) % 3]:
                    feature_values[f"V{i}"] = st.number_input(
                        f"V{i}", value=0.0, step=0.1, format="%.4f",
                        key=f"{key_prefix}_v{i}",
                    )
        submitted = st.form_submit_button(
            "Predict Risk", type="primary", use_container_width=True
        )
    if submitted:
        values = {"Time": time_val, "Amount": amount, **feature_values}
        try:
            score, flagged = predict_values(values)
            st.session_state[f"{key_prefix}_show_result"] = True
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def show_risk_result():
    result = st.session_state.last_prediction
    if not result:
        st.markdown(
            '<div class="risk-card"><div class="risk-title">🛡 Risk Score</div>'
            '<div class="small-note">Submit a transaction to see the model result.</div></div>',
            unsafe_allow_html=True,
        )
        return
    score = result["score"]
    flagged = result["flagged"]
    color = "#ff647c" if flagged else "#37e7b3"
    decision = "Flagged for review" if flagged else "Below threshold"
    st.markdown(
        f"""<div class="risk-card" style="border-color:{color}">
        <div class="risk-title" style="color:{color}">🛡 Risk Score</div>
        <div class="risk-score">{score:.3f}</div>
        <div class="risk-label" style="color:{color}">{decision}</div>
        <div class="small-note">Threshold: {THRESHOLD:.3f} · Amount: {float(result['amount']):,.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("Model score is not necessarily a calibrated fraud probability.")


def section_title(title, subtitle=None):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="panel-subtitle">{subtitle}</div>',
                    unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="font-size:22px;font-weight:800;color:#f5f9ff;">🛡 RiskGuard</div>'
        '<div class="muted" style="margin-top:4px;">Credit Card Fraud Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Single Transaction", "Batch Prediction",
         "Model Performance", "About"],
        label_visibility="visible",
    )
    st.markdown("---")
    if model is not None:
        st.success("Trained model loaded")
        st.caption(f"Model: {MODEL_NAME}")
        st.caption(f"Decision threshold: {THRESHOLD:.3f}")
    else:
        st.warning("Model not loaded")
        st.caption(model_error)
    st.markdown("---")
    st.markdown(
        '<div class="small-note">Educational prototype only. Not a production banking system.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TOP HEADER
# ============================================================
st.markdown(
    f"""
    <div class="topbar">
      <div class="brand">
        <div class="brand-icon">🛡</div>
        <div>
          <div class="brand-title">Credit Card Fraud Detection &amp; Risk Scoring</div>
          <div class="brand-subtitle">Machine Learning&nbsp; • &nbsp;Imbalanced Classification&nbsp; • &nbsp;Streamlit</div>
        </div>
      </div>
      <div class="prototype">Educational Prototype &nbsp; ⓘ</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD: ANALYTICS + RIGHT PREDICTION PANEL
# ============================================================
if page == "Dashboard":
    analytics, right = st.columns([3.45, 1.12], gap="medium")

    with analytics:
        st.markdown(
            '<div class="welcome-title">🛡 Welcome to the Fraud Detection Dashboard</div>'
            '<div class="muted">Analyze transactions, get risk scores, and explore model performance.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        if data is not None and "Class" in data.columns:
            total = len(data)
            fraud = int((data["Class"] == 1).sum())
            legit = int((data["Class"] == 0).sum())
            rate = fraud / total * 100 if total else 0
        else:
            total = fraud = legit = 0
            rate = 0

        m1, m2, m3, m4 = st.columns(4, gap="small")
        with m1:
            metric_card("Total Transactions", f"{total:,}" if data is not None else "—",
                        "From test dataset" if data is not None else "Dataset not found",
                        "blue", "◉")
        with m2:
            metric_card("Fraudulent Transactions", f"{fraud:,}" if data is not None else "—",
                        f"{rate:.3f}% of total" if data is not None else "Class = 1",
                        "red", "⚠")
        with m3:
            metric_card("Legitimate Transactions", f"{legit:,}" if data is not None else "—",
                        f"{100-rate:.2f}% of total" if data is not None else "Class = 0",
                        "green", "✓")
        with m4:
            metric_card("Model Used", MODEL_NAME.replace("_", " ").title(),
                        "Loaded from final_model.joblib" if model is not None else "Model unavailable",
                        "purple", "◎")

        st.write("")
        chart_left, chart_right = st.columns([0.95, 1.65], gap="small")

        with chart_left:
            section_title("Class Distribution")
            if data is not None and "Class" in data.columns:
                counts = data["Class"].map({0: "Legitimate (0)", 1: "Fraud (1)"}).value_counts()
                fig = go.Figure(go.Pie(
                    labels=counts.index, values=counts.values, hole=.68,
                    marker=dict(colors=["#2f8cff", "#ff4568"]),
                    textinfo="none", hovertemplate="%{label}<br>%{value:,}<extra></extra>",
                ))
                fig.add_annotation(
                    text=f"{rate:.2f}%<br><span style='font-size:10px'>Fraud</span>",
                    x=.5, y=.5, showarrow=False, font=dict(size=18, color="#f4f8ff")
                )
                plot_theme(fig, 220)
                fig.update_layout(showlegend=True, legend=dict(orientation="v", x=1.02, y=.5))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.caption(f"Fraud: {fraud:,} · Legitimate: {legit:,}")
            else:
                st.info("Put creditcard.csv in data/raw/ to show class distribution.")
    
        with chart_right:
            section_title("Feature Distributions (Sample)")
            if data is not None and {"Amount", "Class"}.issubset(data.columns):
                sample = data.sample(min(12000, len(data)), random_state=42).copy()
                sample["Transaction type"] = sample["Class"].map({0: "Legit", 1: "Fraud"})
                f1, f2 = st.columns(2, gap="small")
                with f1:
                    st.caption("Amount")
                    fig = px.histogram(
                        sample, x="Amount", color="Transaction type", nbins=55,
                        color_discrete_map={"Legit": "#2f8cff", "Fraud": "#ff4568"},
                        barmode="overlay", opacity=.8,
                    )
                    plot_theme(fig, 205)
                    fig.update_layout(showlegend=True, xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                with f2:
                    st.caption("Time")
                    if "Time" in sample.columns:
                        fig = px.histogram(
                            sample, x="Time", color="Transaction type", nbins=55,
                            color_discrete_map={"Legit": "#2f8cff", "Fraud": "#ff4568"},
                            barmode="overlay", opacity=.8,
                        )
                        plot_theme(fig, 205)
                        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("Time feature not found.")
            else:
                st.info("Dataset needs Amount and Class columns.")
    
        model_report = load_csv_report("model_comparison.csv")
        section_title("Model Comparison", "Validation results · higher is better")
        if model_report is not None and not model_report.empty:
            name_col = next((c for c in ["model", "model_name", "Model"] if c in model_report.columns), None)
            metric_cols = [c for c in ["precision", "recall", "f1", "average_precision"] if c in model_report.columns]
            if name_col and metric_cols:
                melted = model_report.melt(id_vars=name_col, value_vars=metric_cols,
                                           var_name="Metric", value_name="Score")
                fig = px.bar(
                    melted, x=name_col, y="Score", color="Metric", barmode="group",
                    color_discrete_sequence=["#368bff", "#25ce91", "#ffb340", "#a76cff"],
                )
                plot_theme(fig, 235)
                fig.update_layout(xaxis_title="", yaxis_title="", yaxis_range=[0, 1],
                                  legend=dict(orientation="h", y=1.12, x=0))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.dataframe(model_report, use_container_width=True, hide_index=True)
        else:
            st.info("Model comparison report not found. Run `python -m src.train`.")

        pr_path = FIGURES / "08_test_precision_recall_curve.png"
        if pr_path.exists():
            st.markdown('<div class="panel-title">Precision–Recall Curve</div>', unsafe_allow_html=True)
            st.image(str(pr_path), use_container_width=True)

        tabs = st.tabs(["Recent Predictions", "Evaluation Metrics", "Confusion Matrix"])
        with tabs[0]:
            if st.session_state.prediction_history:
                st.dataframe(pd.DataFrame(st.session_state.prediction_history),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("Predictions made in this app session will appear here.")
        with tabs[1]:
            final_metrics = load_csv_report("final_test_metrics.csv")
            if final_metrics is not None:
                st.dataframe(final_metrics, use_container_width=True, hide_index=True)
            else:
                st.caption("Run `python -m src.evaluate` to generate held-out test metrics.")
        with tabs[2]:
            cm_path = FIGURES / "07_test_confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.caption("Confusion matrix image is not available yet.")

    with right:
        section_title("▣ Single Transaction Prediction", "Enter transaction details to get a model score.")
        if model is not None:
            transaction_form("dash")
            show_risk_result()
        else:
            st.error(model_error)

        section_title("Quick Actions")
        if st.button("⇩  Download Batch Results", use_container_width=True):
            st.session_state.quick_nav = "Batch Prediction"
            st.info("Choose Batch Prediction from the sidebar to upload a CSV.")
        if st.button("▥  View Model Comparison", use_container_width=True):
            st.session_state.quick_nav = "Model Performance"
            st.info("Choose Model Performance from the sidebar to inspect metrics.")
        if st.button("◉  Explore Dataset", use_container_width=True):
            if data is not None:
                st.write(f"Dataset rows: {len(data):,} · Columns: {len(data.columns)}")
                st.write("Columns:", ", ".join(data.columns.astype(str)))
            else:
                st.warning("Dataset not found in data/raw/creditcard.csv.")

# ============================================================
# SINGLE TRANSACTION PAGE
# ============================================================
elif page == "Single Transaction":
    st.markdown('<div class="welcome-title">Single Transaction Prediction</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="muted">Score one transaction using the saved model and validation-selected threshold.</div>',
                unsafe_allow_html=True)
    if model is None:
        st.error(model_error)
    else:
        left, right = st.columns([1.4, 1])
        with left:
            transaction_form("single", compact=False)
        with right:
            section_title("Prediction Result")
            show_risk_result()
        st.info("V1–V28 are anonymized features. Use values prepared in the same way as the training data; arbitrary defaults are only for interface testing.")

# ============================================================
# BATCH PREDICTION PAGE
# ============================================================
elif page == "Batch Prediction":
    st.markdown('<div class="welcome-title">Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload a CSV containing the model feature columns and download scored results.</div>',
                unsafe_allow_html=True)
    if model is None:
        st.error(model_error)
    else:
        st.info("Required columns: " + ", ".join(FEATURES))
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            try:
                batch = pd.read_csv(uploaded)
                missing = [c for c in FEATURES if c not in batch.columns]
                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                else:
                    st.dataframe(batch.head(10), use_container_width=True)
                    if st.button("Run Batch Prediction", type="primary"):
                        X = batch[FEATURES].apply(pd.to_numeric, errors="coerce")
                        if X.isna().any().any():
                            st.error("Some required features are missing or non-numeric.")
                        else:
                            scores = model.predict_proba(X)[:, 1]
                            results = batch.copy()
                            results["Model_Score"] = scores
                            results["Flagged_For_Review"] = scores >= THRESHOLD
                            results["Decision"] = np.where(
                                scores >= THRESHOLD, "Flagged for review", "Below threshold"
                            )
                            a, b, c = st.columns(3)
                            a.metric("Rows scored", f"{len(results):,}")
                            b.metric("Flagged for review", f"{int(results['Flagged_For_Review'].sum()):,}")
                            c.metric("Below threshold", f"{int((~results['Flagged_For_Review']).sum()):,}")
                            st.dataframe(results.head(200), use_container_width=True)
                            st.download_button(
                                "Download prediction results",
                                results.to_csv(index=False).encode("utf-8"),
                                "fraud_risk_results.csv", "text/csv",
                                use_container_width=True,
                            )
            except Exception as exc:
                st.error(f"Could not read/process CSV: {exc}")

# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================
elif page == "Model Performance":
    st.markdown('<div class="welcome-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Inspect actual validation comparisons and held-out test evaluation.</div>',
                unsafe_allow_html=True)
    comparison = load_csv_report("model_comparison.csv")
    validation = load_csv_report("validation_model_comparison.csv")
    final_metrics = load_csv_report("final_test_metrics.csv")
    chosen = validation if validation is not None else comparison

    st.subheader("Model comparison")
    if chosen is not None:
        st.dataframe(chosen, use_container_width=True, hide_index=True)
    else:
        st.warning("No comparison report found. Run the training and evaluation scripts.")

    st.subheader("Final test metrics")
    if final_metrics is not None:
        st.dataframe(final_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m src.evaluate` to generate final test metrics.")

    for filename, title in [
        ("06_threshold_optimization.png", "Threshold Optimization"),
        ("07_test_confusion_matrix.png", "Test Confusion Matrix"),
        ("08_test_precision_recall_curve.png", "Test Precision–Recall Curve"),
    ]:
        path = FIGURES / filename
        if path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), use_container_width=True)

# ============================================================
# ABOUT PAGE
# ============================================================
else:
    st.markdown('<div class="welcome-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This educational project demonstrates a machine-learning pipeline for
        credit-card fraud detection on an imbalanced dataset.

        **Workflow**
        1. Load and validate the dataset.
        2. Explore class imbalance and transaction features.
        3. Split data into training, validation and held-out test sets.
        4. Compare Logistic Regression and Random Forest models.
        5. Select a model and threshold using validation data.
        6. Evaluate once on the held-out test set.
        7. Use Streamlit for single and batch scoring.

        **Important limitations**
        - `V1`–`V28` are anonymized dataset features.
        - A model score is not automatically a calibrated probability.
        - A flagged transaction is not proof of fraud.
        - A score below the threshold does not guarantee legitimacy.
        - This app is not a production banking or financial decision system.
        """
    )
    st.subheader("Saved configuration")
    st.write("**Model:**", MODEL_NAME)
    st.write("**Decision threshold:**", f"{THRESHOLD:.6f}")
    st.write("**Feature count:**", len(FEATURES))
    with st.expander("View model metadata"):
        st.json(metadata if metadata else {"status": model_error})
