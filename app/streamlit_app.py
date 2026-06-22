import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartFlow-Aware",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global style ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1d2e;
        padding: 8px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2d4a;
        color: #e0e0e0;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff !important;
        color: #0f1117 !important;
        font-weight: 700;
    }
    .metric-card {
        background-color: #1a1d2e;
        border: 1px solid #3a3d5c;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
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
</style>
""", unsafe_allow_html=True)

COLORS = {
    'primary':   '#00d4ff',
    'secondary': '#ff6b35',
    'accent':    '#7c3aed',
    'positive':  '#10b981',
    'negative':  '#ef4444',
    'neutral':   '#6b7280',
}

plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#b0b0b0',
    'ytick.color':      '#b0b0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d4a',
    'grid.linestyle':   '--',
    'grid.alpha':        0.5,
    'figure.dpi':        100,
})
@st.cache_data
def load_data():
    df = pd.read_csv(
        '../data/processed/smartflow_features.csv',
        index_col='datetime',
        parse_dates=True
    )
    pred_df = pd.read_csv(
        '../data/processed/forecasting_predictions.csv',
        index_col='datetime',
        parse_dates=True
    )
    anomaly_results = pd.read_csv(
        '../data/processed/anomaly_results.csv',
        index_col='Model'
    )
    return df, pred_df, anomaly_results

@st.cache_resource
def load_models():
    lgb_model = joblib.load('../modules/forecasting/lgb_model.pkl')
    scaler_X  = joblib.load('../modules/forecasting/scaler_X.pkl')
    scaler_y  = joblib.load('../modules/forecasting/scaler_y.pkl')
    iso_forest= joblib.load('../modules/anomaly/isolation_forest.pkl')
    vectorizer= joblib.load('../modules/assistant/tfidf_vectorizer.pkl')

    with open('../modules/assistant/summaries.json', 'r') as f:
        data = json.load(f)

    tfidf_matrix = vectorizer.transform(data['summaries'])

    return (lgb_model, scaler_X, scaler_y,
            iso_forest, vectorizer,
            tfidf_matrix, data['summaries'])

df, pred_df, anomaly_results = load_data()
(lgb_model, scaler_X, scaler_y,
 iso_forest, vectorizer,
 tfidf_matrix, summaries) = load_models()

target = 'Global_active_power'

st.sidebar.image(
    "https://img.icons8.com/fluency/96/lightning-bolt.png",
    width=60
)
st.sidebar.title("SmartFlow-Aware")
st.sidebar.markdown("*AI Energy Analytics for Pakistan*")
st.sidebar.divider()

# Groq API key input
groq_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="Free key at console.groq.com"
)

st.sidebar.divider()
st.sidebar.markdown("### Dataset Info")
st.sidebar.metric("Total Hours",    f"{len(df):,}")
st.sidebar.metric("Date Range",
    f"{df.index.min().date()} →\n{df.index.max().date()}")
st.sidebar.metric("Features",       f"{len(df.columns)}")
st.sidebar.metric("Avg Power",
    f"{df[target].mean():.2f} kW")

st.sidebar.divider()
st.sidebar.markdown("### Pakistan Context")

hour_now     = pd.Timestamp.now().hour
is_peak      = 7 <= hour_now < 23
peak_label   = "🔴 Peak Tariff" if is_peak else "🟢 Off-Peak"
month_now    = pd.Timestamp.now().month
is_heat      = month_now in [5, 6, 7, 8]
season_label = "☀️ Heat Season" if is_heat else "🌤️ Moderate"

st.sidebar.info(f"**Tariff:** {peak_label}")
st.sidebar.info(f"**Season:** {season_label}")

st.markdown("""
<h1 style='color:#00d4ff; font-size:2rem; margin-bottom:0'>
    ⚡ SmartFlow-Aware
</h1>
<p style='color:#b0b0b0; margin-top:0; font-size:1rem'>
    AI-Powered Energy Load Forecasting · Anomaly Detection ·
    Intelligent Assistant · Pakistan Context
</p>
<hr style='border-color:#3a3d5c'>
""", unsafe_allow_html=True)

# Top KPI metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg Consumption",
              f"{df[target].mean():.2f} kW",
              f"{((df[target].mean()/1.09)-1)*100:+.1f}% vs baseline")
with col2:
    recent_pred = pred_df['lgb'].tail(24).mean()
    st.metric("24h Forecast (LightGBM)",
              f"{recent_pred:.2f} kW",
              "Next 24 hours")
with col3:
    st.metric("Best Forecaster",
              "LightGBM",
              "MAE: 0.1481 kW")
with col4:
    st.metric("Best Anomaly Detector",
              "Hybrid AE+IF",
              f"ROC-AUC: {anomaly_results.loc['Hybrid AE+IF','ROC_AUC']:.3f}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast",
    "🚨 Anomalies",
    "💬 Assistant",
    "🇵🇰 Pakistan Context"
])

with tab1:
    st.subheader("24-Hour Load Forecasting")
    st.markdown(
        "LightGBM model trained on 42 Pakistan-aware features. "
        "Best performer across MAE, RMSE, and MAPE on held-out test set."
    )

    # Date range selector
    col_a, col_b = st.columns(2)
    with col_a:
        n_days = st.slider(
            "Days to display", 7, 90, 30
        )
    with col_b:
        show_models = st.multiselect(
            "Models to show",
            ['lgb', 'gru', 'bilstm', 'tft'],
            default=['lgb'],
            format_func=lambda x: {
                'lgb':'LightGBM','gru':'GRU',
                'bilstm':'BiLSTM','tft':'TFT (simplified)'
            }[x]
        )

    plot_data = pred_df.tail(n_days * 24)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(plot_data.index, plot_data['actual'],
            color='white', linewidth=1.0,
            alpha=0.8, label='Actual', zorder=3)

    model_colors = {
        'lgb':    COLORS['primary'],
        'gru':    COLORS['accent'],
        'bilstm': COLORS['secondary'],
        'tft':    COLORS['positive']
    }
    model_labels = {
        'lgb':'LightGBM','gru':'GRU',
        'bilstm':'BiLSTM','tft':'TFT'
    }

    for model in show_models:
        ax.plot(plot_data.index, plot_data[model],
                color=model_colors[model],
                linewidth=1.2, alpha=0.85,
                label=model_labels[model])

    ax.set_title(f'Forecast vs Actual — Last {n_days} Days',
                 fontsize=12)
    ax.set_ylabel('Active Power (kW)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Model comparison table
    st.markdown("### Model Comparison — Test Set")
    metrics_data = {
        'Model':    ['LightGBM', 'GRU', 'BiLSTM', 'TFT'],
        'MAE (kW)': [0.1481, 0.4033, 0.4469, 0.5532],
        'RMSE (kW)':[0.2280, 0.5631, 0.6088, 0.7078],
        'MAPE (%)': [18.07,  55.91,  62.39,  82.12],
        'Winner':   ['✓', '', '', '']
    }
    st.dataframe(
        pd.DataFrame(metrics_data).set_index('Model'),
        use_container_width=True
    )

    # Feature importance
    st.markdown("### Top Features — LightGBM")
    importance_df = pd.DataFrame({
        'Feature':   [c for c in df.columns if c != target],
        'Importance': lgb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.barh(importance_df['Feature'][::-1],
             importance_df['Importance'][::-1],
             color=COLORS['primary'], alpha=0.8)
    ax2.set_title('Top 10 LightGBM Feature Importances')
    ax2.set_xlabel('Importance Score')
    ax2.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with tab2:
    st.subheader("Anomaly Detection")
    st.markdown(
        "Four detectors identifying spikes, dropouts, and drift events "
        "in energy consumption. Hybrid AE+IF ensemble achieves best ROC-AUC."
    )

    # Anomaly results table
    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("### Detection Performance")
        st.dataframe(
            anomaly_results.style.highlight_max(
                subset=['ROC_AUC', 'Avg_Precision'],
                color='#1a4a2e'
            ),
            use_container_width=True
        )

    with col_y:
        st.markdown("### Anomaly Types")
        anomaly_types = pd.DataFrame({
            'Type':        ['Spike', 'Dropout', 'Drift'],
            'Description': [
                'Consumption > 3.5x normal',
                'Near-zero during active hours',
                'Gradual sustained increase'
            ],
            'Cause': [
                'Appliance fault / short circuit',
                'Meter tampering / load shedding',
                'Unauthorized connection'
            ],
            'Count': [80, 59, 240]
        }).set_index('Type')
        st.dataframe(anomaly_types, use_container_width=True)

    # Live anomaly detection on selected window
    st.markdown("### Live Detection — Select Time Window")
    window_days = st.slider("Days to analyze", 7, 60, 14)
    window_data = df[[target]].tail(window_days * 24).copy()

    # Build features for anomaly detection
    window_data['rolling_mean'] = (
        window_data[target].rolling(24).mean()
    )
    window_data['z_score'] = (
        (window_data[target] - window_data['rolling_mean'])
        / (window_data[target].rolling(24).std() + 1e-8)
    )
    window_data['rate_of_change'] = (
        window_data[target].diff().abs()
    )
    window_data = window_data.dropna()

    X_window = window_data[[
        target, 'rolling_mean', 'z_score', 'rate_of_change'
    ]].values

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    X_window_sc = sc.fit_transform(X_window)

    iso_live = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42
    )
    iso_live.fit(X_window_sc)
    preds = iso_live.predict(X_window_sc)
    anomaly_mask = preds == -1

    n_detected = anomaly_mask.sum()
    st.metric("Anomalies Detected",
              n_detected,
              f"{n_detected/len(window_data)*100:.1f}% of hours")

    fig3, ax3 = plt.subplots(figsize=(14, 4))
    ax3.plot(window_data.index,
             window_data[target],
             color=COLORS['primary'],
             linewidth=0.8, alpha=0.8, label='Power')

    if anomaly_mask.any():
        ax3.scatter(
            window_data.index[anomaly_mask],
            window_data[target].values[anomaly_mask],
            color=COLORS['negative'],
            s=40, zorder=5, label='Anomaly detected'
        )

    ax3.set_title(
        f'Live Anomaly Detection — Last {window_days} Days '
        f'({n_detected} anomalies found)'
    )
    ax3.set_ylabel('Active Power (kW)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Anomaly table
    if anomaly_mask.any():
        st.markdown("### Detected Anomaly Events")
        anomaly_df = window_data[anomaly_mask][[target, 'z_score']].copy()
        anomaly_df.columns = ['Power (kW)', 'Z-Score']
        anomaly_df['Severity'] = anomaly_df['Z-Score'].abs().apply(
            lambda x: '🔴 High' if x > 3
                      else '🟡 Medium' if x > 2
                      else '🟢 Low'
        )
        st.dataframe(
            anomaly_df.round(3),
            use_container_width=True
        )

with tab3:
    st.subheader("💬 SmartFlow Assistant")
    st.markdown(
        "Ask questions about your energy consumption in plain language. "
        "Powered by Groq/Llama 3.1 with retrieval-augmented generation "
        "and SHAP-based explanations."
    )

    if not groq_key:
        st.warning(
            "⚠️ Enter your Groq API key in the sidebar to use the assistant. "
            "Get a free key at console.groq.com"
        )
    else:
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Intent classifier
        def classify_intent(question: str) -> str:
            q = question.lower()
            scores = {
                'forecast':    sum(k in q for k in
                    ['forecast','predict','tomorrow','next',
                     'will be','expect','future']),
                'anomaly':     sum(k in q for k in
                    ['anomaly','unusual','spike','fault',
                     'problem','abnormal','detected']),
                'explanation': sum(k in q for k in
                    ['why','reason','cause','explain',
                     'because','high','low','bill']),
                'context':     sum(k in q for k in
                    ['load shedding','tariff','temperature',
                     'ramadan','peak','wapda','lesco','heat']),
                'general':     sum(k in q for k in
                    ['average','total','how much','consumption',
                     'usage','summary','statistics']),
            }
            best = max(scores, key=scores.get)
            return best if scores[best] > 0 else 'general'

        # Context retriever
        def retrieve_context(question: str, k: int = 3) -> str:
            query_vec = vectorizer.transform([question])
            scores    = cosine_similarity(
                query_vec, tfidf_matrix
            ).flatten()
            top_idx   = scores.argsort()[-k:][::-1]
            return "\n".join([
                f"[Day {i+1}] {summaries[idx]}"
                for i, idx in enumerate(top_idx)
            ])

        # Ask assistant
        def ask_assistant(question: str) -> str:
            intent  = classify_intent(question)
            context = retrieve_context(question)

            # Build data context by intent
            if intent == 'general':
                data_ctx = (
                    f"Average power: {df[target].mean():.2f} kW. "
                    f"Max: {df[target].max():.2f} kW. "
                    f"Coverage: {df.index.min().date()} to "
                    f"{df.index.max().date()}."
                )
            elif intent == 'forecast':
                recent   = pred_df.tail(24)
                data_ctx = (
                    f"LightGBM 24h forecast: avg {recent['lgb'].mean():.2f} kW, "
                    f"peak {recent['lgb'].max():.2f} kW at "
                    f"{recent['lgb'].idxmax().hour}:00."
                )
            elif intent == 'anomaly':
                data_ctx = (
                    f"Hybrid AE+IF ROC-AUC: "
                    f"{anomaly_results.loc['Hybrid AE+IF','ROC_AUC']:.3f}. "
                    f"Anomaly types: spikes, dropouts, drift."
                )
            elif intent == 'context':
                data_ctx = (
                    f"Peak tariff: {df['is_peak_tariff'].mean()*100:.1f}% of hours. "
                    f"Load shedding: {df['is_load_shedding'].mean()*100:.1f}% of hours. "
                    f"Heat season: {df['is_heat_season'].mean()*100:.1f}% of year."
                )
            else:
                data_ctx = (
                    f"Top SHAP features: Sub_metering_3, lag_1h, "
                    f"Sub_metering_2, Sub_metering_1, Voltage."
                )

            system_prompt = """You are SmartFlow Assistant, an AI energy
advisor for Pakistani households using SkyElectric's SmartFlow system.
You explain electricity consumption, anomalies, forecasts, and bills
in simple language. You know Pakistan's WAPDA tariff schedules, LESCO
load shedding, Punjab heat seasons, and Ramadan patterns. Be concise,
specific, and data-driven."""

            client  = Groq(api_key=groq_key)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-6:]
            ]

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history,
                    {"role": "user", "content":
                        f"{question}\n\nData:\n{context}\n\n{data_ctx}"}
                ],
                temperature=0.3,
                max_tokens=350
            )
            return response.choices[0].message.content

        # Display chat history
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.markdown(
                    f"<div class='chat-user'>"
                    f"<b>You:</b> {msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-assistant'>"
                    f"<b>SmartFlow:</b> {msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Suggested questions
        st.markdown("**Suggested questions:**")
        suggestions = [
            "What is my average daily consumption?",
            "Why is my bill high in summer?",
            "How does load shedding affect my usage?",
            "What time should I avoid heavy appliances?",
            "Were any anomalies detected?",
        ]

        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(suggestion,
                             use_container_width=True,
                             key=f"btn_{suggestion[:20]}"):
                    st.session_state.pending_question = suggestion

        # Text input
        user_input = st.chat_input(
            "Ask about your energy consumption..."
        )

        # Handle input from button or text
        question = None
        if user_input:
            question = user_input
        elif hasattr(st.session_state, 'pending_question'):
            question = st.session_state.pending_question
            del st.session_state.pending_question

        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.spinner("SmartFlow is thinking..."):
                answer = ask_assistant(question)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            st.rerun()

        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear conversation"):
                st.session_state.messages = []
                st.rerun()

with tab4:
    st.subheader("🇵🇰 Pakistan Energy Context")
    st.markdown(
        "Pakistan-specific features engineered on top of the UCI dataset. "
        "These features are the core research contribution of SmartFlow-Aware."
    )

    # Live context indicators
    st.markdown("### Current Status")
    c1, c2, c3, c4 = st.columns(4)

    hour_now  = pd.Timestamp.now().hour
    month_now = pd.Timestamp.now().month

    with c1:
        is_peak = 7 <= hour_now < 23
        st.metric(
            "WAPDA Tariff",
            "Peak" if is_peak else "Off-Peak",
            f"Hour {hour_now}:00"
        )
    with c2:
        is_heat = month_now in [5,6,7,8]
        st.metric(
            "Season",
            "Heat Season" if is_heat else "Moderate",
            "May–August = AC surge"
        )
    with c3:
        st.metric(
            "Load Shedding Rate",
            "16.7%",
            "4h rotation per day"
        )
    with c4:
        st.metric(
            "Peak Tariff Hours",
            "66.7%",
            "07:00–23:00 daily"
        )

    st.divider()

    # Feature distribution plots
    st.markdown("### Pakistan Feature Analysis")

    fig4, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig4.suptitle(
        'Pakistan Context Features — Statistical Overview',
        fontsize=13
    )

    # Plot 1: Binary feature active rates
    binary_features = [
        'is_peak_tariff', 'is_load_shedding', 'is_heat_season',
        'is_extreme_heat', 'is_ramadan', 'is_friday', 'is_weekend'
    ]
    binary_means = df[binary_features].mean() * 100
    bar_colors   = [COLORS['primary']] * len(binary_features)

    axes[0,0].barh(binary_features, binary_means,
                   color=bar_colors, alpha=0.8)
    axes[0,0].set_title('Binary Features — % Hours Active')
    axes[0,0].set_xlabel('Percentage (%)')
    axes[0,0].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(binary_means):
        axes[0,0].text(v+0.3, i, f'{v:.1f}%',
                       va='center', fontsize=8)

    # Plot 2: Temperature distribution
    axes[0,1].hist(df['temperature_c'], bins=50,
                   color=COLORS['secondary'], alpha=0.8,
                   edgecolor='none')
    axes[0,1].axvline(40, color=COLORS['negative'],
                      linestyle='--',
                      label='Extreme heat (40°C)')
    axes[0,1].set_title('Punjab Temperature Distribution')
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].legend(fontsize=9)
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Consumption — peak vs off-peak
    peak_data    = df[df['is_peak_tariff']==1][target]
    offpeak_data = df[df['is_peak_tariff']==0][target]
    axes[1,0].hist(offpeak_data, bins=50, alpha=0.7,
                   color=COLORS['neutral'],
                   label='Off-Peak', edgecolor='none')
    axes[1,0].hist(peak_data, bins=50, alpha=0.7,
                   color=COLORS['secondary'],
                   label='Peak Tariff', edgecolor='none')
    axes[1,0].set_title('Power: Peak vs Off-Peak Hours')
    axes[1,0].set_xlabel('Active Power (kW)')
    axes[1,0].legend(fontsize=9)
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Monthly consumption pattern
    monthly_avg = df.groupby('month')[target].mean()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    bar_c = [COLORS['secondary'] if m in [5,6,7,8]
             else COLORS['primary']
             for m in range(1,13)]
    axes[1,1].bar(month_names, monthly_avg.values,
                  color=bar_c, alpha=0.85,
                  edgecolor='white', linewidth=0.3)
    axes[1,1].set_title('Monthly Average Consumption')
    axes[1,1].set_ylabel('Avg Power (kW)')
    axes[1,1].grid(True, alpha=0.3, axis='y')

    legend_elements = [
        mpatches.Patch(facecolor=COLORS['secondary'],
                       label='Heat Season (May–Aug)'),
        mpatches.Patch(facecolor=COLORS['primary'],
                       label='Other Months'),
    ]
    axes[1,1].legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # Pakistan feature engineering table
    st.markdown("### Feature Engineering Summary")
    feat_table = pd.DataFrame({
        'Feature':     [
            'is_peak_tariff', 'is_load_shedding',
            'temperature_c', 'is_extreme_heat',
            'is_heat_season', 'is_ramadan',
            'is_friday', 'peak_x_heat',
            'heat_x_shedding', 'ramadan_evening'
        ],
        'Source': [
            'NEPRA/WAPDA schedule',
            'LESCO feeder rotation',
            'Punjab climate profile',
            'temperature > 40°C',
            'Months 5–8',
            'Islamic calendar',
            'Day of week == 4',
            'Interaction feature',
            'Interaction feature',
            'Interaction feature'
        ],
        'Active %': [
            f"{df['is_peak_tariff'].mean()*100:.1f}%",
            f"{df['is_load_shedding'].mean()*100:.1f}%",
            f"{df['temperature_c'].mean():.1f}°C avg",
            f"{df['is_extreme_heat'].mean()*100:.1f}%",
            f"{df['is_heat_season'].mean()*100:.1f}%",
            f"{df['is_ramadan'].mean()*100:.1f}%",
            f"{df['is_friday'].mean()*100:.1f}%",
            f"{df['peak_x_heat'].mean()*100:.1f}%",
            f"{df['heat_x_shedding'].mean()*100:.1f}%",
            f"{df['ramadan_evening'].mean()*100:.1f}%",
        ]
    }).set_index('Feature')

    st.dataframe(feat_table, use_container_width=True)