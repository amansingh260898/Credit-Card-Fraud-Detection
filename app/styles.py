
def load_css():
    return """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1120 0%, #101827 55%, #111827 100%);
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    .hero {
        padding: 28px;
        border: 1px solid #334155;
        border-radius: 20px;
        background: linear-gradient(120deg, #172554, #172033 70%);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 8px;
        color: #f8fafc;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .metric-card {
        background: #172033;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px;
        min-height: 105px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 6px;
    }

    .risk-card {
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #334155;
        background: #172033;
        margin-top: 12px;
        margin-bottom: 16px;
    }

    .risk-low {
        border-left: 5px solid #22c55e;
    }

    .risk-high {
        border-left: 5px solid #ef4444;
    }

    .risk-review {
        border-left: 5px solid #f59e0b;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """
