
import streamlit as st


def render_metric(label, value, help_text=None):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if help_text:
        st.caption(help_text)


def render_risk_result(risk_score, is_fraud, threshold):
    if is_fraud:
        risk_label = "Flagged for review"
        risk_class = "risk-high"
        icon = "🚨"
        message = (
            "The model score meets or exceeds the configured "
            "fraud threshold. Review this transaction."
        )
    else:
        risk_label = "Below fraud threshold"
        risk_class = "risk-low"
        icon = "🛡️"
        message = (
            "The model score is below the configured threshold. "
            "This does not guarantee that the transaction is legitimate."
        )

    st.markdown(
        f"""
        <div class="risk-card {risk_class}">
            <h3>{icon} {risk_label}</h3>
            <p>Model fraud score: <b>{risk_score:.2f}%</b></p>
            <p>Decision threshold: <b>{threshold * 100:.2f}%</b></p>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
