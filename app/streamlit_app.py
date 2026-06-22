import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from groq import Groq
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Absolute path resolution ──────────────────────────────────────────────
# Works both locally (app/ subfolder) and on Streamlit Cloud (repo root)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODULES_DIR = os.path.join(BASE_DIR, "modules")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartFlow-Aware",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .main { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background-color: #12172a; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #1a1d2e;
        padding: 8px 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2d4a;
        color: #b0b0b0;
        border-radius: 6px;
        padding: 8px 18px;
        font-size: 0.875rem;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff !important;
        color: #0f1117 !important;
        font-weight: 700;
    }

    /* Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #12172a 100%);
        border: 1px solid #2a2d4a;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .kpi-label {
        color: #6b7280;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #00d4ff;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-delta {
        color: #10b981;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Chat bubbles */
    .chat-user {
        background-color: #1a1d2e;
        border-left: 3px solid #00d4ff;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
        color: #f1f5f9;
    }
    .chat-assistant {
        background-color: #12172a;
        border-left: 3px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
        color: #e2e8f0;
    }

    /* Section headings */
    .section-label {
        color: #6b7280;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    /* Anomaly badge */
    .badge-high   { color: #ef4444; font-weight: 700; }
    .badge-medium { color: #ff6b35; font-weight: 700; }
    .badge-low    { color: #10b981; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#2a2d4a",
    "axes.labelcolor":  "#b0b0b0",
    "xtick.color":      "#6b7280",
    "ytick.color":      "#6b7280",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "figure.dpi":       110,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

C = {
    "primary":   "#00d4ff",
    "secondary": "#ff6b35",
    "accent":    "#7c3aed",
    "positive":  "#10b981",
    "negative":  "#ef4444",
    "neutral":   "#6b7280",
    "warning":   "#f59e0b",
}

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        os.path.join(DATA_DIR, "smartflow_features.csv"),
        index_col="datetime", parse_dates=True,
    )
    pred_df = pd.read_csv(
        os.path.join(DATA_DIR, "forecasting_predictions.csv"),
        index_col="datetime", parse_dates=True,
    )
    anomaly_results = pd.read_csv(
        os.path.join(DATA_DIR, "anomaly_results.csv"),
        index_col="Model",
    )
    return df, pred_df, anomaly_results


@st.cache_resource
def load_models():
    lgb_model  = joblib.load(os.path.join(MODULES_DIR, "forecasting", "lgb_model.pkl"))
    scaler_X   = joblib.load(os.path.join(MODULES_DIR, "forecasting", "scaler_X.pkl"))
    scaler_y   = joblib.load(os.path.join(MODULES_DIR, "forecasting", "scaler_y.pkl"))
    iso_forest = joblib.load(os.path.join(MODULES_DIR, "anomaly",     "isolation_forest.pkl"))
    vectorizer = joblib.load(os.path.join(MODULES_DIR, "assistant",   "tfidf_vectorizer.pkl"))

    with open(os.path.join(MODULES_DIR, "assistant", "summaries.json"), "r") as f:
        data = json.load(f)

    tfidf_matrix = vectorizer.transform(data["summaries"])
    return lgb_model, scaler_X, scaler_y, iso_forest, vectorizer, tfidf_matrix, data["summaries"]


df, pred_df, anomaly_results = load_data()
lgb_model, scaler_X, scaler_y, iso_forest, vectorizer, tfidf_matrix, summaries = load_models()
TARGET = "Global_active_power"

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 4px'>
        <span style='font-size:2.2rem'>⚡</span><br>
        <span style='color:#00d4ff; font-size:1.1rem; font-weight:700'>SmartFlow-Aware</span><br>
        <span style='color:#6b7280; font-size:0.75rem'>AI Energy Analytics · Pakistan</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com — required for the Assistant tab",
    )

    st.divider()
    st.markdown("<div class='section-label'>Dataset</div>", unsafe_allow_html=True)
    st.metric("Hours",      f"{len(df):,}")
    st.metric("Avg Power",  f"{df[TARGET].mean():.2f} kW")
    st.metric("Features",   f"{len(df.columns)}")
    st.metric("Date Range",
              f"{df.index.min().strftime('%b %Y')} – {df.index.max().strftime('%b %Y')}")

    st.divider()
    st.markdown("<div class='section-label'>Live Pakistan Status</div>",
                unsafe_allow_html=True)

    hour_now  = pd.Timestamp.now().hour
    month_now = pd.Timestamp.now().month
    is_peak   = 7 <= hour_now < 23
    is_heat   = month_now in [5, 6, 7, 8]

    st.info(f"{'🔴 Peak Tariff' if is_peak else '🟢 Off-Peak'} — {hour_now:02d}:00")
    st.info(f"{'☀️ Heat Season' if is_heat else '🌤️ Moderate Season'}")

    st.divider()
    st.markdown(
        "<div style='color:#3a3d5c; font-size:0.7rem; text-align:center'>"
        "Built for Ashar Aziz Center for AI · GIKI<br>"
        "github.com/Umar-Shahid/smartflow-aware"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 4px'>
    <h1 style='color:#00d4ff; font-size:1.9rem; margin:0; font-weight:800;
               letter-spacing:-0.02em'>
        ⚡ SmartFlow-Aware
    </h1>
    <p style='color:#6b7280; margin:4px 0 0; font-size:0.9rem'>
        Load Forecasting &nbsp;·&nbsp; Anomaly Detection &nbsp;·&nbsp;
        Intelligent Assistant &nbsp;·&nbsp; Pakistan Energy Context
    </p>
</div>
""", unsafe_allow_html=True)

# KPI row
k1, k2, k3, k4 = st.columns(4)
kpi_data = [
    ("Best Forecaster",  "LightGBM",   "MAE 0.1481 kW"),
    ("24h Avg Forecast", f"{pred_df['lgb'].tail(24).mean():.2f} kW",
     "Next 24 hours"),
    ("Best Detector",    "Hybrid AE+IF",
     f"ROC-AUC {anomaly_results.loc['Hybrid AE+IF','ROC_AUC']:.3f}"),
    ("Pakistan Features","10 engineered",
     "Tariff · Load shedding · Ramadan"),
]
for col, (label, value, delta) in zip([k1, k2, k3, k4], kpi_data):
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-delta'>{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Forecast",
    "🚨  Anomalies",
    "💬  Assistant",
    "🇵🇰  Pakistan Context",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("24-Hour Load Forecasting")
    st.markdown(
        "LightGBM trained on 42 Pakistan-aware features — best performer "
        "across MAE, RMSE, and MAPE on a chronological held-out test set."
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_days = st.slider("Days to display", 7, 90, 30)
    with col_b:
        show_models = st.multiselect(
            "Models to overlay",
            options=["lgb", "gru", "bilstm", "tft"],
            default=["lgb"],
            format_func=lambda x: {
                "lgb":    "LightGBM",
                "gru":    "GRU",
                "bilstm": "BiLSTM",
                "tft":    "TFT (simplified)",
            }[x],
        )

    plot_data = pred_df.tail(n_days * 24)
    model_colors = {
        "lgb":    C["primary"],
        "gru":    C["accent"],
        "bilstm": C["secondary"],
        "tft":    C["positive"],
    }
    model_labels = {
        "lgb": "LightGBM", "gru": "GRU",
        "bilstm": "BiLSTM", "tft": "TFT",
    }

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(plot_data.index, plot_data["actual"],
            color="white", linewidth=0.9, alpha=0.75,
            label="Actual", zorder=3)
    for m in show_models:
        ax.plot(plot_data.index, plot_data[m],
                color=model_colors[m], linewidth=1.2,
                alpha=0.9, label=model_labels[m])
    ax.set_title(f"Forecast vs Actual — Last {n_days} Days", fontsize=12)
    ax.set_ylabel("Active Power (kW)")
    ax.legend(fontsize=9, loc="upper right",
              facecolor="#1a1d2e", edgecolor="#2a2d4a")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Model Comparison — Test Set")
    results_df = pd.DataFrame({
        "Model":     ["LightGBM", "GRU", "BiLSTM", "TFT (simplified)"],
        "MAE (kW)":  [0.1481, 0.4033, 0.4469, 0.5532],
        "RMSE (kW)": [0.2280, 0.5631, 0.6088, 0.7078],
        "MAPE (%)":  [18.07,  55.91,  62.39,  82.12],
        "Rank":      ["🥇 1st", "2nd", "3rd", "4th"],
    }).set_index("Model")
    st.dataframe(results_df, use_container_width=True)

    st.markdown("### Top 10 Feature Importances — LightGBM")
    feat_cols = [c for c in df.columns if c != TARGET]
    imp_df = pd.DataFrame({
        "Feature":    feat_cols,
        "Importance": lgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bars = ax2.barh(
        imp_df["Feature"].values[::-1],
        imp_df["Importance"].values[::-1],
        color=C["primary"], alpha=0.85,
    )
    ax2.set_xlabel("Importance Score")
    ax2.set_title("LightGBM — Top 10 Feature Importances")
    ax2.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALIES
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Anomaly Detection")
    st.markdown(
        "Four detectors — Isolation Forest, LSTM Autoencoder, Hybrid AE+IF, "
        "and One-Class SVM — identifying spikes, dropouts, and drift events."
    )

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("### Detection Performance")
        st.dataframe(
            anomaly_results.style.highlight_max(
                subset=["ROC_AUC", "Avg_Precision"],
                color="#1a3a2e",
            ),
            use_container_width=True,
        )

    with col_y:
        st.markdown("### Anomaly Types")
        at_df = pd.DataFrame({
            "Type":        ["Spike", "Dropout", "Drift"],
            "Description": [
                "Consumption > 3.5× normal",
                "Near-zero during active hours",
                "Gradual sustained increase",
            ],
            "Likely Cause": [
                "Appliance fault / short circuit",
                "Meter tampering / load shedding",
                "Unauthorized connection",
            ],
            "Count": [80, 59, 240],
        }).set_index("Type")
        st.dataframe(at_df, use_container_width=True)

    st.divider()
    st.markdown("### Live Detection — Select Time Window")

    window_days = st.slider("Days to analyse", 7, 60, 14)
    window_data = df[[TARGET]].tail(window_days * 24).copy()

    window_data["rolling_mean"] = window_data[TARGET].rolling(24).mean()
    window_data["rolling_std"]  = window_data[TARGET].rolling(24).std()
    window_data["z_score"]      = (
        (window_data[TARGET] - window_data["rolling_mean"])
        / (window_data["rolling_std"] + 1e-8)
    )
    window_data["rate_of_change"] = window_data[TARGET].diff().abs()
    window_data = window_data.dropna()

    X_w  = window_data[[TARGET, "rolling_mean", "z_score", "rate_of_change"]].values
    sc   = MinMaxScaler()
    X_ws = sc.fit_transform(X_w)

    iso_live = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso_live.fit(X_ws)
    live_preds    = iso_live.predict(X_ws)
    anomaly_mask  = live_preds == -1
    n_detected    = anomaly_mask.sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Hours Analysed", f"{len(window_data):,}")
    m2.metric("Anomalies Detected", n_detected)
    m3.metric("Anomaly Rate", f"{n_detected / len(window_data) * 100:.1f}%")

    fig3, ax3 = plt.subplots(figsize=(14, 4))
    ax3.plot(window_data.index, window_data[TARGET],
             color=C["primary"], linewidth=0.8, alpha=0.8, label="Power")
    if anomaly_mask.any():
        ax3.scatter(
            window_data.index[anomaly_mask],
            window_data[TARGET].values[anomaly_mask],
            color=C["negative"], s=45, zorder=5,
            label=f"Anomaly ({n_detected} detected)", marker="^",
        )
    ax3.set_title(f"Live Anomaly Detection — Last {window_days} Days")
    ax3.set_ylabel("Active Power (kW)")
    ax3.legend(fontsize=9, facecolor="#1a1d2e", edgecolor="#2a2d4a")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    if anomaly_mask.any():
        st.markdown("### Detected Anomaly Events")
        anom_df = window_data[anomaly_mask][[TARGET, "z_score"]].copy()
        anom_df.columns = ["Power (kW)", "Z-Score"]
        anom_df["Severity"] = anom_df["Z-Score"].abs().apply(
            lambda x: "🔴 High" if x > 3 else "🟡 Medium" if x > 2 else "🟢 Low"
        )
        st.dataframe(anom_df.round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — ASSISTANT
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("💬 SmartFlow Assistant")
    st.markdown(
        "Ask about your energy consumption in plain language. "
        "Powered by Groq / Llama 3.1 with TF-IDF retrieval-augmented generation."
    )

    if not groq_key:
        st.warning(
            "⚠️ Enter your Groq API key in the sidebar to activate the assistant. "
            "Get a free key at **console.groq.com**."
        )
        st.stop()

    # ── Helper functions ──────────────────────────────────────────────────
    def classify_intent(question: str) -> str:
        q = question.lower()
        scores = {
            "forecast":    sum(k in q for k in
                ["forecast", "predict", "tomorrow", "next",
                 "will be", "expect", "future", "upcoming"]),
            "anomaly":     sum(k in q for k in
                ["anomaly", "unusual", "spike", "fault",
                 "problem", "abnormal", "detected", "strange"]),
            "explanation": sum(k in q for k in
                ["why", "reason", "cause", "explain",
                 "because", "high", "low", "bill"]),
            "context":     sum(k in q for k in
                ["load shedding", "tariff", "temperature",
                 "ramadan", "peak", "wapda", "lesco", "heat", "weather"]),
            "general":     sum(k in q for k in
                ["average", "total", "how much", "consumption",
                 "usage", "summary", "overview", "statistics"]),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def retrieve_context(question: str, k: int = 3) -> str:
        query_vec = vectorizer.transform([question])
        scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_idx   = scores.argsort()[-k:][::-1]
        return "\n".join(
            f"[Day {i+1}] {summaries[idx]}"
            for i, idx in enumerate(top_idx)
        )

    def build_data_context(intent: str) -> str:
        if intent == "general":
            return (
                f"Average power: {df[TARGET].mean():.2f} kW. "
                f"Max: {df[TARGET].max():.2f} kW. "
                f"Coverage: {df.index.min().date()} to {df.index.max().date()}."
            )
        if intent == "forecast":
            r = pred_df.tail(24)
            return (
                f"LightGBM 24h forecast: avg {r['lgb'].mean():.2f} kW, "
                f"peak {r['lgb'].max():.2f} kW at {r['lgb'].idxmax().hour}:00."
            )
        if intent == "anomaly":
            return (
                f"Hybrid AE+IF ROC-AUC: "
                f"{anomaly_results.loc['Hybrid AE+IF','ROC_AUC']:.3f}. "
                f"Anomaly types detected: spikes, dropouts, drift."
            )
        if intent == "context":
            return (
                f"Peak tariff active {df['is_peak_tariff'].mean()*100:.1f}% of hours. "
                f"Load shedding {df['is_load_shedding'].mean()*100:.1f}% of hours. "
                f"Heat season {df['is_heat_season'].mean()*100:.1f}% of year. "
                f"Extreme heat {df['is_extreme_heat'].mean()*100:.1f}% of hours."
            )
        return (
            "Top SHAP features: Sub_metering_3, lag_1h, "
            "Sub_metering_2, Sub_metering_1, Voltage."
        )

    SYSTEM_PROMPT = (
        "You are SmartFlow Assistant, an AI energy advisor for Pakistani "
        "households using SkyElectric's SmartFlow system. You explain "
        "electricity consumption, anomalies, forecasts, and bills in simple "
        "language. You know Pakistan's WAPDA/NEPRA tariff schedules (peak "
        "07:00–23:00), LESCO load shedding rotation, Punjab heat seasons "
        "(May–August, 40°C+), and Ramadan consumption shifts. Be concise, "
        "specific, and data-driven. Answers should be 3–5 sentences unless "
        "the user asks for detail."
    )

    def ask_assistant(question: str) -> str:
        intent   = classify_intent(question)
        context  = retrieve_context(question)
        data_ctx = build_data_context(intent)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-6:]
        ]

        client   = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
                {
                    "role": "user",
                    "content": (
                        f"{question}\n\n"
                        f"Relevant energy records:\n{context}\n\n"
                        f"Key stats: {data_ctx}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=350,
        )
        return response.choices[0].message.content

    # ── Chat UI ───────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested questions
    st.markdown("**Try asking:**")
    suggestions = [
        "What is my average daily consumption?",
        "Why is my bill high in summer?",
        "How does load shedding affect usage?",
        "When should I avoid heavy appliances?",
        "Were any anomalies detected?",
    ]
    s_cols = st.columns(len(suggestions))
    for col, suggestion in zip(s_cols, suggestions):
        with col:
            if st.button(suggestion, use_container_width=True,
                         key=f"sug_{suggestion[:15]}"):
                st.session_state.pending = suggestion

    st.markdown("<br>", unsafe_allow_html=True)

    # Display history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-user'><b>You:</b> {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-bot'><b>⚡ SmartFlow:</b> {msg['content']}</div>",
                unsafe_allow_html=True,
            )

    # Input
    user_input = st.chat_input("Ask about your energy consumption...")

    question = None
    if user_input:
        question = user_input
    elif hasattr(st.session_state, "pending"):
        question = st.session_state.pending
        del st.session_state.pending

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer = ask_assistant(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — PAKISTAN CONTEXT
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🇵🇰 Pakistan Energy Context")
    st.markdown(
        "Pakistan-specific features engineered on top of the UCI dataset — "
        "the core research contribution of SmartFlow-Aware."
    )

    # Live status
    st.markdown("### Current Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WAPDA Tariff",
              "Peak" if is_peak else "Off-Peak",
              f"Hour {hour_now:02d}:00")
    c2.metric("Season",
              "Heat Season" if is_heat else "Moderate",
              "May–Aug = AC surge")
    c3.metric("Load Shedding Rate", "16.7%", "4h rotation per day")
    c4.metric("Peak Tariff Coverage", "66.7%", "07:00–23:00 daily")

    st.divider()

    # Four analysis plots
    fig4, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig4.suptitle("Pakistan Context Features — Overview", fontsize=13, y=1.01)

    # 1 — Binary features
    bin_feats = [
        "is_peak_tariff", "is_load_shedding", "is_heat_season",
        "is_extreme_heat", "is_ramadan", "is_friday", "is_weekend",
    ]
    bin_means = df[bin_feats].mean() * 100
    axes[0, 0].barh(bin_feats, bin_means, color=C["primary"], alpha=0.85)
    axes[0, 0].set_title("Binary Features — % Hours Active")
    axes[0, 0].set_xlabel("Percentage (%)")
    axes[0, 0].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(bin_means):
        axes[0, 0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    # 2 — Temperature distribution
    axes[0, 1].hist(df["temperature_c"], bins=50,
                    color=C["secondary"], alpha=0.85, edgecolor="none")
    axes[0, 1].axvline(40, color=C["negative"], linestyle="--",
                       linewidth=1.5, label="Extreme heat (40°C)")
    axes[0, 1].set_title("Punjab Temperature Distribution")
    axes[0, 1].set_xlabel("Temperature (°C)")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # 3 — Peak vs off-peak power
    pk  = df[df["is_peak_tariff"] == 1][TARGET]
    opk = df[df["is_peak_tariff"] == 0][TARGET]
    axes[1, 0].hist(opk, bins=50, alpha=0.7, color=C["neutral"],
                    label="Off-Peak", edgecolor="none")
    axes[1, 0].hist(pk,  bins=50, alpha=0.7, color=C["secondary"],
                    label="Peak Tariff", edgecolor="none")
    axes[1, 0].set_title("Power Distribution: Peak vs Off-Peak")
    axes[1, 0].set_xlabel("Active Power (kW)")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 4 — Monthly average
    monthly_avg = df.groupby("month")[TARGET].mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    bar_colors  = [C["secondary"] if m in [5,6,7,8] else C["primary"]
                   for m in range(1, 13)]
    axes[1, 1].bar(month_names, monthly_avg.values,
                   color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.3)
    axes[1, 1].set_title("Monthly Average Consumption")
    axes[1, 1].set_ylabel("Avg Power (kW)")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    legend_els = [
        mpatches.Patch(facecolor=C["secondary"], label="Heat Season (May–Aug)"),
        mpatches.Patch(facecolor=C["primary"],   label="Other Months"),
    ]
    axes[1, 1].legend(handles=legend_els, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # Feature engineering table
    st.markdown("### Pakistan Feature Engineering")
    feat_table = pd.DataFrame({
        "Feature": [
            "is_peak_tariff", "is_load_shedding", "temperature_c",
            "is_extreme_heat", "is_heat_season", "is_winter_season",
            "is_ramadan", "is_friday", "peak_x_heat",
            "heat_x_shedding", "ramadan_evening",
        ],
        "Source": [
            "NEPRA/WAPDA schedule", "LESCO feeder rotation",
            "Punjab climate profile", "temperature > 40°C",
            "Months 5–8", "Months 12, 1, 2",
            "Islamic calendar", "Day of week == 4",
            "Interaction: tariff × heat", "Interaction: heat × shedding",
            "Interaction: Ramadan × 19–21h",
        ],
        "Active %": [
            f"{df['is_peak_tariff'].mean()*100:.1f}%",
            f"{df['is_load_shedding'].mean()*100:.1f}%",
            f"{df['temperature_c'].mean():.1f}°C avg",
            f"{df['is_extreme_heat'].mean()*100:.1f}%",
            f"{df['is_heat_season'].mean()*100:.1f}%",
            f"{df['is_winter_season'].mean()*100:.1f}%",
            f"{df['is_ramadan'].mean()*100:.1f}%",
            f"{df['is_friday'].mean()*100:.1f}%",
            f"{df['peak_x_heat'].mean()*100:.1f}%",
            f"{df['heat_x_shedding'].mean()*100:.1f}%",
            f"{df['ramadan_evening'].mean()*100:.1f}%",
        ],
    }).set_index("Feature")
    st.dataframe(feat_table, use_container_width=True)