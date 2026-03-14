import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from predict_live_FINAL import LivePredictionEngine
import datetime
import time

# --- PREMIUM STYLING ---
st.set_page_config(
    page_title="Riq's NBA Elite Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, dark-mode premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .value-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #3d3d5c;
    }
    .highlight {
        color: #00ffcc;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- APP HEADER ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://www.nba.com/assets/logos/nba-logo.svg", width=80)
with col2:
    st.title("🏀 Riq's NBA Elite Predictor")
    st.markdown("*Institutional-Grade Ensemble V4 Engine*")

st.markdown("---")

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("🎛️ Command Center")
    target_date = st.date_input("Target Date", datetime.date.today())
    
    st.markdown("### 🛠️ Maintenance")
    if st.button("🔄 Sync NBA Data", use_container_width=True):
        with st.status("Syncing with NBA API...", expanded=True) as status:
            from daily_refresh import daily_refresh
            st.write("Fetching latest player logs...")
            if daily_refresh():
                status.update(label="Sync Complete!", state="complete", expanded=False)
                st.balloons()
            else:
                status.update(label="Sync Failed.", state="error")

    st.markdown("### ⚙️ Engine Settings")
    min_ev = st.slider("Min Expected Value (EV)", -0.5, 0.5, 0.05, 0.01)
    min_win_prob = st.slider("Min Win Probability", 0.4, 0.8, 0.55, 0.01)

# --- DATA LOADING ---
@st.cache_resource
def load_engine():
    return LivePredictionEngine(
        models_dir="models",
        aggregated_data_path="final_feature_matrix_with_per_min_1997_onward.csv"
    )

with st.spinner("🧠 Waking up the Ensemble Neural Engine..."):
    try:
        engine = load_engine()
        st.sidebar.caption("✅ Model V4 Loaded (XGB + TabNet)")
    except Exception as e:
        st.error(f"Engine Load Failed: {e}")
        st.stop()

# --- MAIN DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["💎 Value Picks", "📊 All Projections", "📈 Performance"])

with tab1:
    st.header("Top +EV Opportunities")
    
    if st.button("🚀 Generate Picks", type="primary", use_container_width=True):
        with st.spinner("Simulating 10,000 game outcomes..."):
            date_str = target_date.strftime("%Y-%m-%d")
            preds = engine.predict_all_games(date=date_str)
            
            if preds.empty:
                st.warning("No games found in roster for this date.")
            else:
                lines = engine.fetch_betting_lines(date=date_str)
                if lines:
                    opportunities = engine.find_ev_opportunities(preds.to_dict('records'), lines)
                    if opportunities:
                        df_ev = pd.DataFrame(opportunities)
                        
                        # High-level metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Opportunities", len(df_ev))
                        m2.metric("Best Edge", f"{df_ev['expected_value'].max():.1%}")
                        m3.metric("Avg Win Prob", f"{df_ev['win_probability'].mean():.1%}")
                        
                        # Filter by user sliders
                        df_filtered = df_ev[(df_ev['expected_value'] >= min_ev) & 
                                            (df_ev['win_probability'] >= min_win_prob)]
                        
                        # Fancy Display for filtered picks
                        if not df_filtered.empty:
                            st.markdown(f"### 🔥 Top Qualified Picks ({len(df_filtered)})")
                            for _, pick in df_filtered.head(10).iterrows():
                                with st.container():
                                    st.markdown(f"""
                                    <div class="value-card">
                                        <h4>{pick['player']} - <span class="highlight">{pick['prop_type'].upper()} {pick['pick']} {pick['line']}</span></h4>
                                        <p><b>Bookmaker:</b> {pick['bookmaker'].upper()} | <b>Odds:</b> {pick.get('odds', 'N/A')}</p>
                                        <progress value="{pick['win_probability']}" max="1"></progress>
                                        <p><b>Win Prob:</b> {pick['win_probability']:.1%} | <b>EV:</b> {pick['expected_value']:+.2f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No picks match your current filters. Try lowering the sliders!")
                    else:
                        st.info("No value found today. Markets are efficient.")
                else:
                    st.error("Odds API returned empty or key error.")

with tab2:
    st.header("Full Player Slate")
    if 'preds' in locals() and not preds.empty:
        st.dataframe(preds, use_container_width=True)
        
        # Distribution Chart
        st.subheader("Projection Distribution")
        fig = px.histogram(preds, x="points", nbins=20, title="Points Projection Spread",
                           color_discrete_sequence=['#00ffcc'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Generate Picks' in the Value Picks tab to populate this data.")

with tab3:
    st.header("Engine Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Freshness")
        st.write(f"**Last Sync Date:** {engine.aggregated_data['gameDate'].max()}")
        st.write(f"**Total Records:** {len(engine.aggregated_data):,}")
    with c2:
        st.subheader("Model Status")
        st.info("✅ Ensemble Layer: Online")
        st.info("✅ Shapley Explainer: Ready")

st.markdown("---")
st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
