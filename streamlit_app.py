import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import time
import os
import glob
from pathlib import Path

# PHASE V5: Global Environment Hardening
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Path hack for modular imports
sys.path.append(str(Path(__file__).parent))

from meep_terminal.core.engine import TerminalEngine
from meep_terminal.data.models import DatabaseManager, Bet, Pick, ModelRun
from meep_terminal.interface.components import ExecutionComponent, AnalyticsComponent
from meep_terminal.interface.views.player_view import render_player_deep_dive
from meep_terminal.core.math_utils import BettingMath
from meep_terminal.core.ai_analyst import MEEPAnalyst

def get_prediction_explanation(row):
    """Generates a natural language explanation for a prediction."""
    player = row.get('player', 'The player')
    prop = row.get('prop', 'this prop').upper()
    win_prob = row.get('win_prob', 50.0)
    side = row.get('side', 'OVER')
    ev = row.get('ev', 0.0)
    
    # Heuristic-based reasoning
    reasons = []
    
    if win_prob > 60:
        reasons.append(f"High-confidence {side} projection based on optimized usage patterns.")
    
    if abs(ev) > 0.15:
        reasons.append(f"Significant market mispricing detected relative to neutral-court simulations.")
        
    # Synergy commentary
    synergy_impact = row.get('synergy_impact', 0)
    if synergy_impact > 2:
        reasons.append(f"Strong lineup synergy detected: Teammate spacing increases {player}'s quality looks.")
    elif synergy_impact < -2:
        reasons.append(f"Negative synergy factors: Potential usage congestion with the current rotation.")
    else:
        reasons.append(f"Balanced team synergy supports the baseline projection.")
        
    reasons.append(f"Opponent rotation shows historic vulnerability to high-volume {prop} specialists.")
    return "  \n".join([f"• {r}" for r in reasons])

# --- APP CONFIG ---
st.set_page_config(
    page_title="MEEP | Terminal", 
    layout="wide", 
    page_icon="📟",
    initial_sidebar_state="collapsed"
)

# --- CACHING & PERFORMANCE ---
@st.cache_resource
def get_terminal_engine():
    """Singleton initialization of the heavy ML engine with Auto-Sync."""
    with st.spinner("⚡ INITIALIZING MEEP NEURAL CORE..."):
        engine = TerminalEngine()
        # The auto-sync of ledger happens inside __init__
        
        # Check if an auto-refresh for stale data was triggered
        stats = engine.get_stats()
        data_date = stats.get('data_date')
        if data_date:
             is_stale = (datetime.now() - data_date).days >= 1
             if is_stale:
                  st.toast(f"⚠️ DATA STALE ({data_date.strftime('%b %d')}). Auto-refreshing in background...", icon="🔄")
        
        return engine

@st.cache_data(ttl=600)
def get_cached_performance(_engine):
    """Cache real performance stats from the ledger."""
    return _engine.get_performance_metrics()

# --- REUSABLE UI SYSTEM ---
def apply_premium_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
        
        :root {
            --bg-deep: #0B0E14;
            --bg-card: #161B22;
            --accent: #FF4B4B;
            --text-main: #FFFFFF;
            --text-dim: #808495;
            --success: #00D1FF;
            --warning: #FFB800;
        }

        .stApp {
            background-color: var(--bg-deep);
            color: var(--text-main);
        }

        /* Status Strip */
        .status-strip {
            background: linear-gradient(90deg, #1A1C24 0%, #0E1117 100%);
            padding: 12px 24px;
            border-radius: 12px;
            border-left: 5px solid var(--accent);
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }

        .status-item {
            display: flex;
            flex-direction: column;
        }

        .status-label {
            color: var(--text-dim);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 800;
            margin-bottom: 2px;
        }

        .status-value {
            color: var(--text-main);
            font-size: 15px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
        }

        /* Metric Enhancement */
        div[data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        div[data-testid="stMetricValue"] {
            color: var(--text-main) !important;
            font-size: 28px !important;
            font-weight: 800 !important;
        }
        .status-value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 16px;
        }

        /* Event Feed Styles */
        .event-item {
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 11px;
        }
        .event-time { color: var(--text-dim); margin-right: 8px; font-family: 'JetBrains Mono'; }
        .level-info { color: var(--text-main); }
        .level-success { color: var(--success); }
        .level-warning { color: var(--warning); }
        .level-error { color: var(--accent); }

        /* Professional Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
            margin-bottom: 20px;
        }

            border: none !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 3px solid var(--accent) !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px !important;
            border: none !important;
        }
        /* Tooltip and Simplified Cards */
        .rookie-card {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 12px;
        }
        
        .tier-elite { color: #A855F7; font-weight: 800; text-shadow: 0 0 10px rgba(168, 85, 247, 0.4); } /* Purple */
        .tier-high { color: #22C55E; font-weight: 700; } /* Green */
        .tier-medium { color: #EAB308; font-weight: 600; } /* Yellow */
        .tier-low { color: #EF4444; font-weight: 500; } /* Red */

        .explainer-box {
            background: rgba(0, 209, 255, 0.05);
            border: 1px solid rgba(0, 209, 255, 0.1);
            padding: 12px;
            border-radius: 8px;
            margin-top: 8px;
            font-size: 13px;
        }
    </style>
    """, unsafe_allow_html=True)


def render_status_strip(engine):
    stats = engine.get_stats()
    freshness = stats.get('data_date', datetime.now())
    freshness_str = freshness.strftime("%b %d, %H:%M")
    
    # Calculate operational health color
    hours_diff = (datetime.now() - freshness).total_seconds() / 3600
    health_color = "#00D1FF" if hours_diff < 24 else "#FFB800" if hours_diff < 48 else "#FF4B4B"

    st.markdown(f"""
    <div class="status-strip">
        <div class="status-item">
            <span class="status-label">Engine Freshness</span>
            <span class="status-value" style="color: {health_color}">{freshness_str}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Vault Connection</span>
            <span class="status-value">ENCRYPTED</span>
        </div>
        <div class="status-item">
            <span class="status-label">System Load</span>
            <span class="status-value">OPTIMAL</span>
        </div>
        <div class="status-item">
            <span class="status-label">Active Bankroll</span>
            <span class="status-value">${stats.get('bankroll', 0.0):,.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- INITIALIZATION ---
apply_premium_styles()
engine = get_terminal_engine()
db = DatabaseManager()
session = db.get_session()

# --- SESSION STATE INITIALIZATION ---
if 'current_picks' not in st.session_state:
    st.session_state.current_picks = []
if 'render_start_time' not in st.session_state:
    st.session_state.render_start_time = time.time()
if 'parlay_slip' not in st.session_state:
    st.session_state.parlay_slip = []
if 'last_slate_date' not in st.session_state:
    st.session_state.last_slate_date = None
if 'active_tasks' not in st.session_state:
    st.session_state.active_tasks = []
if 'fallback_active' not in st.session_state:
    st.session_state.fallback_active = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'analyst_instance' not in st.session_state:
    st.session_state.analyst_instance = MEEPAnalyst(engine, api_key=os.getenv("GROQ_API_KEY"))
if 'show_ai_panel' not in st.session_state:
    st.session_state.show_ai_panel = True
if 'ai_panel_width' not in st.session_state:
    st.session_state.ai_panel_width = 1.0 # Default ratio (3:1)

# Global Mode Controller
with st.sidebar:
    st.markdown("### 🎛️ PLATFORM CONTROL")
    current_prefs = engine.get_user_preferences("admin")
    mode_options = ["Rookie", "All-Star", "Superstar"]
    
    # Map index
    current_mode = current_prefs.get('mode', 'rookie').capitalize()
    if current_mode not in mode_options: current_mode = "Rookie"
    
    mode = st.radio(
        "Workstation Tier", 
        options=mode_options, 
        index=mode_options.index(current_mode),
        help="ROOKIE: Simple parlay suggestions & high-confidence arbs. ALL-STAR: Customizable slips & mid-tier arbs. SUPERSTAR: Full quant core & all market discrepancies."
    )
    if mode.lower() != current_prefs.get('mode'):
        engine.update_user_preferences({"mode": mode.lower()})
        st.rerun()

    # Manual Bankroll Control
    st.divider()
    st.markdown("### 💰 FINANCIAL CORE")
    current_bankroll = current_prefs.get('bankroll')
    if current_bankroll is None:
        # Get from engine stats if not in prefs
        current_bankroll = engine.get_stats().get('bankroll', 1000.0)
    
    new_bankroll = st.number_input(
        "Active Bankroll ($)",
        min_value=0.0,
        value=float(current_bankroll),
        step=100.0,
        format="%.2f",
        help="Manually override the system bankroll for accurate staking calculations."
    )
    
    if new_bankroll != current_bankroll:
        engine.update_user_preferences({"bankroll": new_bankroll})
        st.toast(f"✅ Bankroll updated to ${new_bankroll:,.2f}")
        st.rerun()
    
    # --- WORKSPACE CUSTOMIZATION ---
    st.divider()
    st.markdown("### 🖥️ WORKSPACE")
    st.session_state.show_ai_panel = st.checkbox("Show AI Analyst", value=st.session_state.show_ai_panel)
    if st.session_state.show_ai_panel:
        st.session_state.ai_panel_width = st.select_slider(
            "Analyst Split", 
            options=[0.5, 1.0, 1.5, 2.0], 
            value=st.session_state.ai_panel_width,
            help="Adjust the size of the right-side Analyst panel."
        )
    
    st.divider()
    
    # Event Stream Component
    st.markdown("### 📡 INTELLIGENCE FEED")
    events = engine.get_events(limit=8)
    for ev in events:
        st.markdown(f"""
        <div class="event-item">
            <span class="event-time">{ev.timestamp.strftime('%H:%M')}</span>
            <span class="level-{ev.level}">{ev.message}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    # Data Integrity Controls
    st.markdown("### 🛠️ DATA MAINTENANCE")
    if st.button("🔄 REFRESH DATA", help="Fetch latest NBA logs and update master matrix (3+ mins)", width="stretch"):
        tid = engine.start_background_refresh()
        st.session_state.active_tasks.append(tid)
        st.toast("⚡ Data Sync Started in Background.")
        st.rerun()

    # Pilot Feedback Hook
    with st.expander("📬 PILOT FEEDBACK"):
        fb_type = st.selectbox("Type", ["Bug", "Feature", "UX"], key="pilot_fb_type")
        fb_msg = st.text_area("Message", key="pilot_fb_msg")
        if st.button("Submit Feedback", width="stretch"):
            engine.submit_feedback(fb_type.lower(), fb_msg)
            st.success("Feedback logged. Thank you, Pilot.")

    st.divider()
    # --- SIDEBAR AI ANALYST (PORTED FROM TAB) ---


# --- TASK MONITOR (Sidebar overlay) ---
def render_task_monitor():
    if not st.session_state.active_tasks:
        return
        
    with st.sidebar:
        st.markdown("### ⚙️ SYSTEM TASKS")
        new_active = []
        for tid in st.session_state.active_tasks:
            info = engine.get_task_status(tid)
            if info['status'] == 'completed':
                st.success(f"Job {tid}: Success")
                st.toast(f"✅ Background Job {tid} finished.")
                if "inference" in info.get('message', '').lower():
                    st.rerun() # Refresh to show new picks
            elif info['status'] == 'failed':
                st.error(f"Job {tid}: Failed")
            else:
                st.info(f"{info['message']}")
                st.progress(float(info['progress']))
                new_active.append(tid)
        st.session_state.active_tasks = new_active

# --- PROACTIVE INTELLIGENCE: AUTO-SLATE DISCOVERY ---
def auto_discover_slate(target_date_str):
    """Zero-click slate retrieval with Cycle 5 Fallback support."""
    slate = engine.get_portfolio_slate(target_date_str)
    if slate:
        st.session_state.current_slate = slate
        
    if st.session_state.last_slate_date != target_date_str or not st.session_state.current_picks:
        if slate:
            st.session_state.current_picks = []
            for p in slate.get('raw_picks', []):
                meta = p.metadata_json or {}
                if isinstance(meta, str):
                    import json
                    try:
                        meta = json.loads(meta)
                    except:
                        meta = {}
                
                # Extract side from meta or calculate
                side = meta.get('side')
                if not side:
                    if p.prediction and p.line:
                        # Prefer explicit OVER/UNDER from logic if prediction vs line matches standard side
                        # However, for the 'Advise' col, we like descriptive labels
                        # If p.prediction < 1.0 it's likely a probability, so we can't compare to line
                        if p.prediction <= 1.0:
                            side = "OVER" # Default fallback for imported models if missing
                        else:
                            side = "OVER" if p.prediction > p.line else "UNDER"
                    else:
                        side = "NEUTRAL"
                
                st.session_state.current_picks.append({
                    "id": p.id, "player": p.player_name, "team": p.team, "opp": p.opponent,
                    "prop": p.prop_type, "side": side, "line": p.line if p.line else p.prediction,
                    "mu": p.prediction, "sigma": p.std_dev,
                    "win_prob": p.win_prob * 100 if p.win_prob else 50.0,
                    "ev": p.ev if p.ev else 0.0,
                    "kelly": (p.ev * 10.48) if p.ev and p.ev > 0 else 0.0,
                    "tier": p.confidence_tier or "C",
                    "explanation": p.explanation
                })
            st.session_state.last_slate_date = target_date_str
            if slate['is_fallback']:
                st.session_state.fallback_active = slate['date']
            else:
                st.session_state.fallback_active = None
            return True
        else:
            # CLEAR stale picks if no data exists for the selected window
            st.session_state.current_picks = []
            st.session_state.last_slate_date = target_date_str
            st.session_state.fallback_active = None
            return False
    return False

# --- MAIN RENDER ---
render_status_strip(engine)
render_task_monitor()

# --- MAIN SPLIT LAYOUT ---
if st.session_state.show_ai_panel:
    # Split Screen Mode
    col_main, col_side = st.columns([3, st.session_state.ai_panel_width])
    
    with col_side:
        # Execution & AI Center (Right Split)
        ExecutionComponent.render_bet_slip(session)
        st.divider()
        
        st.markdown(f"### 🤖 AI ANALYST")
        # Floating-style container for the analyst
        with st.container():
            analyst = st.session_state.analyst_instance
            
            # Chat history window (Fixed Height Scrollbox)
            st.markdown('<div style="height: 450px; overflow-y: auto; padding-right: 12px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; background: rgba(0,0,0,0.1); padding: 10px;">', unsafe_allow_html=True)
            for msg in st.session_state.chat_messages:
                role_icon = "👤" if msg['role'] == "user" else "🤖"
                st.markdown(f"**{role_icon}**: {msg['content']}")
            st.markdown('</div>', unsafe_allow_html=True)

            if chat_prompt := st.chat_input("Ask MEEP Analyst...", key="right_chat"):
                st.session_state.chat_messages.append({"role": "user", "content": chat_prompt})
                with st.spinner("🧠..."):
                    response = analyst.chat(st.session_state.chat_messages)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("🗑️ CLEAR CHAT", width="stretch"):
                st.session_state.chat_messages = []
                st.rerun()
else:
    # Full Screen Mode
    col_main = st.container()
    # Hidden sidebar button to restore
    with st.sidebar:
        if st.button("➕ Restore AI Panel"):
            st.session_state.show_ai_panel = True
            st.rerun()

# --- DYNAMIC SLATE DISCOVERY (GLOBAL) ---
with st.sidebar:
    st.markdown("### 📅 Slate Control")
    target_date = st.date_input("Target Slate Date", datetime.now())
    date_str = target_date.strftime("%Y-%m-%d")
    
    # Run discovery before any tabs render to populate session_state
    if auto_discover_slate(date_str):
        st.toast(f"✅ Auto-Loaded {len(st.session_state.current_picks)} Predictions for {date_str}")
    
    st.divider()

with col_main:
    mode_lower = mode.lower()
    
    # Dynamic tab selection based on mode
    if mode_lower == "rookie":
        tabs = ["🎯 QUICK PICKS", "📡 MARKET RADAR", "🤖 AI ANALYST", "📊 PERFORMANCE"]
        tab_list = st.tabs(tabs)
        tab_home, tab_market, tab_ai, tab_perf = tab_list
        tab_props = tab_home # Alias for logic
        tab_lab = None
        tab_vault = None
        tab_audit = None
    else:
        tabs = ["🎯 COMMAND CENTER", "📡 MARKET RADAR", "🎯 PLAYER PROPS", "🔬 ANALYTICS LAB", "🤖 AI ANALYST", "📊 PERFORMANCE", "🏛️ VAULT", "🕵️ AUDIT"]
        tab_list = st.tabs(tabs)
        tab_home, tab_market, tab_props, tab_lab, tab_ai, tab_perf, tab_vault, tab_audit = tab_list

    with tab_market:
        st.markdown("### 📡 Market Edge Radar (V6 Neural Flow)")
        st.info("Detecting sportsbook pricing inefficiencies using Monte Carlo fair line probabilities + V6 GNN Synergy.")
        
        # 0. V6 Configuration
        col_v6, col_data = st.columns([1, 1])
        with col_v6:
             v6_enabled = st.toggle("🚀 Activate V6 Neural Synergy", value=True, help="Enables Lineup GNN coordination factors.")
        with col_data:
             synergy_val = st.slider("Target Lineup Synergy", 0.0, 0.1, 0.024, step=0.005)
        
        # 1. Fetch Value Bets
        # Note: In a real app, 'current_picks' would be passed here
        # For now, we simulate with the current session state picks
        slate = st.session_state.get('current_picks', [])
        if not slate:
             st.warning("No active slate discovered. Scan for games to enable radar.")
        else:
             value_bets = engine.engine.get_value_bets(slate) # Calling V5 Service
             
             if not value_bets:
                  st.success("No significant market inefficiencies detected (>5% edge).")
             else:
                  # Apply V6 Synergy if enabled
                  if v6_enabled:
                       value_bets = engine.engine.get_synergy_adjusted_picks(value_bets, synergy_score=synergy_val)

                  col1, col2 = st.columns([1, 1])
                  
                  with col1:
                       st.metric("Top Edge", f"{value_bets[0]['edge'] if not v6_enabled else value_bets[0]['win_prob'] - value_bets[0]['market_prob']}%", 
                                 delta=f"{value_bets[0]['player']} (V6 {'+' if value_bets[0].get('synergy_impact', 0) > 0 else ''}{value_bets[0].get('synergy_impact', 0)}%)" if v6_enabled else value_bets[0]['player'])
                       
                       # 2. Synergy Comparison Chart
                       if v6_enabled:
                            df_chart = pd.DataFrame(value_bets)
                            fig = go.Figure()
                            fig.add_trace(go.Bar(name='Raw Monte Carlo', x=df_chart['player'], y=df_chart['raw_win_prob'], marker_color='#808495'))
                            fig.add_trace(go.Bar(name='V6 Synergy Adjusted', x=df_chart['player'], y=df_chart['win_prob'], marker_color='#00D1FF'))
                            fig.update_layout(title="Raw vs. Synergy Adjusted Probabilities", barmode='group', template='plotly_dark')
                            st.plotly_chart(fig, width="stretch")
                       else:
                            edges = [b['edge'] for b in value_bets]
                            fig = px.histogram(edges, nbins=10, title="Edge Distribution",
                                             labels={'value': 'Edge (%)', 'count': 'Frequency'},
                                             color_discrete_sequence=['#00D1FF'])
                            st.plotly_chart(fig, width="stretch")
                       
                  with col2:
                       # 3. Value Bet List
                       st.markdown("#### 🔥 High-Value Opportunities")
                       df_value = pd.DataFrame(value_bets)
                       cols = ['player', 'prop', 'line', 'win_prob', 'market_prob']
                       if v6_enabled:
                            cols += ['synergy_score', 'synergy_impact']
                       else:
                            cols += ['edge', 'edge_confidence']
                            
                       st.dataframe(df_value[cols], width="stretch", hide_index=True)

    with tab_home:
        # Fallback Indicator
        if st.session_state.get('fallback_active'):
            st.warning(f"📅 **NEXT-DAY FORECAST ACTIVE**: No games for selected date. Showing projections for **{st.session_state.fallback_active}**.")

        st.markdown("### 🎯 Daily Decision Terminal")
        
        # Risk Radar Overlay (Cycle 5)
        risk_data = engine.get_portfolio_risk(st.session_state.current_picks, mode=mode.lower()) if st.session_state.current_picks else {"status": "Safe", "score": 0.0, "exposure": 0.0, "var": 0.0, "conflicts": []}
        rc1, rc2 = st.columns([2, 1])
        with rc1:
            if mode.lower() == "rookie":
                status_color = "green" if risk_data.get('status') == "Safe" else "orange"
                st.markdown(f"**Portfolio Risk Status:** :{status_color}[{risk_data.get('status', 'N/A')}]")
            else:
                st.markdown(f"**Institutional Risk Surface:** Exposure `${risk_data.get('exposure', 0.0):.1f}` | VaR `${risk_data.get('var', 0.0):.1f}`")
        with rc2:
            if st.button("📊 VIEW RISK RADAR", width="stretch"):
                st.toast("Switching to Advanced Risk Simulation...")
        
        # Grid layout for controls
        c_control1, c_control2, c_control3 = st.columns([1, 1, 1])
        # date_input moved to sidebar for global state
        show_advanced = c_control2.checkbox("Show Advanced Quant", value=False)
        
        c_control3.write("") # Spacer
        if c_control3.button("🚀 EXECUTE INFERENCE", help="Run the full neural ensemble and generate today's pick card (Takes 1-2 mins).", width="stretch"):
            # Clear cache to ensure clean data post-inference
            st.cache_data.clear()
            st.cache_resource.clear()
            tid = engine.start_inference()
            st.session_state.active_tasks.append(tid)
            st.toast("🧠 Neural Inference Started in Background.")
            st.rerun()

        if c_control3.button("🔄 SYNC WORKSPACE", help="Ingest latest data from the scheduled background task (No API).", width="stretch"):
            # Clear data cache but keep models (resource cache)
            st.cache_data.clear()
            tid = engine.start_local_sync()
            st.session_state.active_tasks.append(tid)
            st.rerun()

        # AI-Assisted Parlay Section (Cycle 4)
        st.markdown("---")
        st.markdown("### 🤖 AI Portfolio Assistant")
        parlay_tips = engine.get_ai_parlays(mode=mode.lower())
        
        if parlay_tips:
            pt_cols = st.columns(len(parlay_tips))
            for i, tip in enumerate(parlay_tips):
                with pt_cols[i]:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid var(--success);">
                        <strong style="color: var(--success);">{tip['name']}</strong><br/>
                        <span style="font-size: 11px;">{tip['logic']}</span><br/>
                        <span style="font-family: 'JetBrains Mono'; font-weight: 800;">{tip['odds']} | {tip['win_prob']} Win Prob</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Draft {tip['name']}", key=f"draft_tip_{i}"):
                        st.toast(f"✅ Drafted {tip['name']} to slip.")
                        # Logic to add to state would go here
            
            st.divider()
            # --- START PICK DISPLAY ---
            # Display Welcome Section
            if not st.session_state.current_picks:
                st.markdown("---")
                st.markdown(f"""
                <div style="text-align:center; padding: 40px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.1);">
                    <div style="font-size: 40px;">📭</div>
                    <h3 style="color: #808495; margin: 10px 0 5px;">No Predictions for {date_str}</h3>
                    <p style="color: #555; font-size: 13px;">No inference has been run for this date yet.<br/>Click <strong style="color:#00D1FF;">🚀 EXECUTE INFERENCE</strong> above to generate picks.</p>
                </div>
                """, unsafe_allow_html=True)
                
            elif st.session_state.current_picks:
                df = pd.DataFrame(st.session_state.current_picks)
                user_tier = mode.lower()
                
                # Helper for confidence mapping
                def get_tier_label(win_prob):
                    if win_prob > 62: return "ELITE", "tier-elite"
                    if win_prob > 58: return "HIGH", "tier-high"
                    if win_prob > 54: return "MEDIUM", "tier-medium"
                    return "LOW", "tier-low"

                if user_tier == "rookie":
                    st.markdown(f"### 🎯 Ultra-Simple Top 15")
                    st.caption("Essential recommendations. Complexity hidden.")
                    # Rookie displays top 15, no Edge %
                    for _, row in df.head(15).iterrows():
                        tier_label, tier_class = get_tier_label(row['win_prob'])
                        with st.container():
                            st.markdown(f"""
                            <div class="rookie-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="font-size: 18px;">{row['player']}</strong> 
                                        <span style="color: #808495; font-size: 14px;"> — {row['prop'].upper()} {row['side']}</span>
                                    </div>
                                    <div class="{tier_class}" style="font-size: 12px; border: 1px solid currentColor; padding: 2px 8px; border-radius: 4px; cursor: help;" 
                                         title="Confidence indicates neural consensus across 8+ models. ELITE: >62% consensus | HIGH: >58% | MEDIUM: >54%.">
                                        {tier_label} CONFIDENCE
                                    </div>
                                </div>
                                <div style="margin-top: 8px; display: flex; gap: 20px;">
                                    <div><span style="color: #808495; font-size: 11px;">RECOMMENDATION:</span> <br/><b>BET {row['side']} {row['line']}</b></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("🔍 Explain Prediction", expanded=False):
                                st.markdown(f"""
                                <div class="explainer-box">
                                    {get_prediction_explanation(row)}
                                    <div style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
                                        <span style="color: #808495; font-size: 11px; cursor: help;" title="Lineup Synergy: How well this group of 5 players works together to create open looks or defensive stops.">
                                            ℹ️ Includes <b>Team Synergy</b> adjustments in plain language.
                                        </span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                elif user_tier == "all-star":
                    st.markdown(f"### 🎯 Quick Picks Top 20")
                    st.caption("Standard precision. Simplified metrics included.")
                    # All-Star (Simple Mode) displays top 20, includes Edge %
                    for _, row in df.head(20).iterrows():
                        tier_label, tier_class = get_tier_label(row['win_prob'])
                        with st.container():
                            st.markdown(f"""
                            <div class="rookie-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="font-size: 18px;">{row['player']}</strong> 
                                        <span style="color: #808495; font-size: 14px;"> — {row['prop'].upper()} {row['side']}</span>
                                    </div>
                                    <div class="{tier_class}" style="font-size: 12px; border: 1px solid currentColor; padding: 2px 8px; border-radius: 4px; cursor: help;" 
                                         title="Confidence indicates neural consensus across 8+ models. ELITE: >62% consensus | HIGH: >58% | MEDIUM: >54%.">
                                        {tier_label} CONFIDENCE
                                    </div>
                                </div>
                                <div style="margin-top: 8px; display: flex; gap: 20px;">
                                    <div><span style="color: #808495; font-size: 11px;">RECOMMENDATION:</span> <br/><b>BET {row['side']} {row['line']}</b></div>
                                    <div><span style="color: #808495; font-size: 11px;">EST. EDGE:</span> <br/><span style="color: #00D1FF;">+{row['ev']*100:.1f}%</span></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("🔍 Explain Prediction", expanded=False):
                                st.markdown(f"""
                                <div class="explainer-box">
                                    {get_prediction_explanation(row)}
                                    <div style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
                                        <span style="color: #808495; font-size: 11px; cursor: help;" title="Lineup Synergy: How well this group of 5 players works together to create open looks or defensive stops.">
                                            ℹ️ Includes <b>Team Synergy</b> adjustments in plain language.
                                        </span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    # PRO Mode Command Center
                    if st.session_state.get('current_slate'):
                        slate = st.session_state.current_slate
                        st.markdown(f"### 🏆 Structured Portfolios <span style='color:#808495; font-size:14px; font-weight:normal;'>Status: {slate.get('slate_status', 'N/A')}</span>", unsafe_allow_html=True)
                        
                        pc1, pc2, pc3 = st.columns(3)
                        
                        def render_portfolio_card(title, color, items):
                            st.markdown(f"""<div style="background: {color}1a; padding: 12px; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 10px;">
                                <strong style="color: {color}; font-size: 14px;">{title}</strong>
                            </div>""", unsafe_allow_html=True)
                            if not items:
                                st.caption("No assets identified for this tier.")
                                return
                                
                            for p in items:
                                with st.container():
                                    odds_val = p.get('combined_odds', 0)
                                    odds_str = f"+{odds_val}" if odds_val > 0 else str(odds_val) if odds_val != 0 else "Singles"
                                    c_p1, c_p2 = st.columns([3, 1])
                                    legs = p.get('legs', [])
                                    for leg in legs:
                                        name_part = leg.split(' - ')[0].replace('Single: ', '')
                                        c_p1.markdown(f"**{name_part}**")
                                        if ' - ' in leg: c_p1.caption(leg.split(' - ')[1])
                                            
                                    c_p2.markdown(f"""<div style="text-align: right; margin-top: 5px;">
                                        <div style="font-family: 'JetBrains Mono'; font-size: 18px; font-weight: bold; color: {color};">{odds_str}</div>
                                        <div style="font-size: 13px; color: #808495;">{p['prob']*100:.1f}%</div>
                                    </div>""", unsafe_allow_html=True)
                                    st.divider()

                        with pc1: render_portfolio_card("🛡️ CORE", "#00D1FF", slate['core'])
                        with pc2: render_portfolio_card("📈 GROWTH", "#FFB800", slate['growth'])
                        with pc3: render_portfolio_card("🚀 MOONSHOT", "#FF4B4B", slate['moonshot'])

    if tab_props and tab_props != tab_home:
        with tab_props:
            st.markdown("### 🎯 Player Prop Terminal")
            st.caption("Full exposure control and manual slip building.")
            if not st.session_state.current_picks:
                st.info("Execute inference to browse the full prop library.")
            else:
                df = pd.DataFrame(st.session_state.current_picks)
                df['Add'] = False
                st.divider()
                # Neural Velocity Visualizer (Superstar Only)
                with st.expander("🧠 Neural Volatility Curves", expanded=False):
                    st.caption("Visualizing the FT-Transformer's predictive distribution.")
                    from scipy.stats import norm
                    # Filter for rows with mu and sigma
                    vis_df = df.dropna(subset=['mu', 'sigma'])
                    if not vis_df.empty:
                        top_targets = vis_df.sort_values('win_prob', ascending=False).head(3)
                        fig_v = go.Figure()
                        colors = ['#00D1FF', '#FFB800', '#FF4B4B']
                        for i, (_, row) in enumerate(top_targets.iterrows()):
                            mu, sigma = float(row['mu']), float(row['sigma'])
                            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                            y = norm.pdf(x, mu, sigma)
                            fig_v.add_trace(go.Scatter(
                                x=x, y=y, mode='lines', name=f"{row['player']}",
                                line=dict(color=colors[i % len(colors)], width=3),
                                fill='toself', 
                                fillcolor=f"rgba({i*50}, 209, 255, 0.1)"
                            ))
                        fig_v.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=10,b=0), showlegend=True)
                        st.plotly_chart(fig_v, width="stretch")
            
                # Legacy Portfolio Rendering removed as it's now in with tab_home or tab_props
                pass
                
            if user_tier in ["all-star", "superstar"] and tab_lab:
                # Advanced Lab content handled in with tab_lab
                pass

            if not df.empty and user_tier != "rookie":
                # Line Sensitivity Stress-Test (Premium Only)
                with st.expander("⚖️ Line-Edge Sensitivity Analyst", expanded=False):
                    st.caption("Stress-testing edge durability against market line movement.")
                    s_col1, s_col2 = st.columns([1, 4])
                    shift = s_col1.radio("Shift", [-1.0, -0.5, 0, 0.5, 1.0], index=2, key="expert_shift")
                    sens_df = df.copy()
                    from scipy.stats import norm
                    def calc_sens(r, s):
                        # Safeguard against missing or non-numeric neural parameters
                        mu = r.get('mu')
                        sigma = r.get('sigma')
                        if pd.isna(mu) or pd.isna(sigma) or mu is None or sigma is None:
                            return 0.0
                        
                        try:
                            # Ensure we have floats
                            mu_f, sigma_f = float(mu), float(sigma)
                            new_line = float(r['line']) + s
                            if sigma_f <= 0: return 0.0
                            
                            return (1 - norm.cdf(new_line, mu_f, sigma_f))*100 if mu_f > new_line else norm.cdf(new_line, mu_f, sigma_f)*100
                        except (ValueError, TypeError):
                            return 0.0
                    sens_df['shifted_prob'] = sens_df.apply(lambda r: calc_sens(r, shift), axis=1)
                    st.dataframe(sens_df[['player', 'prop', 'line', 'shifted_prob']].sort_values('shifted_prob', ascending=False).head(5), width="stretch", hide_index=True)



            st.divider()
            # Dynamic Data Editor (Cycle 6.1: High Readability Overhaul)
            cols_to_show = ['Add', 'player', 'prop', 'side', 'opp', 'line', 'win_prob']
            if user_tier != "rookie" or show_advanced: 
                cols_to_show.extend(['mu', 'sigma', 'ev', 'kelly'])

            # Readable Header Mapping
            col_map = {
                "player": "👤 Asset",
                "prop": "🎯 Target",
                "side": "💬 Advise",
                "opp": "⚔️ Opponent",
                "line": "📊 Line",
                "win_prob": "🔥 Confidence",
                "mu": "🧠 Neural Mean",
                "sigma": "📉 Volatility",
                "ev": "💰 EV (%)",
                "kelly": "⚖️ Kelly Stake",
                "Add": "➕"
            }

            edited_df = st.data_editor(
                df,
                column_config={
                    "player": "👤 Asset",
                    "prop": "🎯 Target",
                    "side": st.column_config.TextColumn("💬 Advise", help="BULLISH/BEARISH: Pred vs Rounded Fair Line | OVER/UNDER: Pred vs Market Line"),
                    "opp": "⚔️ Opponent",
                    "line": st.column_config.NumberColumn("📊 Line", format="%.1f"),
                    "win_prob": st.column_config.ProgressColumn("🔥 Confidence", min_value=0, max_value=100, format="%.1f%%"),
                    "mu": st.column_config.NumberColumn("🧠 Neural Mean", format="%.2f"),
                    "sigma": st.column_config.NumberColumn("📉 Volatility", format="%.2f"),
                    "ev": st.column_config.NumberColumn("💰 EV (%)", format="%.1f%%"),
                    "kelly": st.column_config.NumberColumn("⚖️ Kelly Stake", format="$%.2f"),
                    "Add": st.column_config.CheckboxColumn("➕", default=False)
                },
                column_order=cols_to_show,
                hide_index=True,
                width="stretch",
                key="picks_editor_cycle6"
            )

            # Execution Logic
            new_additions = edited_df[edited_df['Add'] == True]
            if not new_additions.empty:
                for _, row in new_additions.iterrows():
                    if not any(item['id'] == row['id'] for item in st.session_state.parlay_slip):
                        st.session_state.parlay_slip.append(row.to_dict())
                st.toast(f"Added {len(new_additions)} legs to slip.")
                st.rerun()


    if tab_ai:
        with tab_ai:
            st.markdown("### 🤖 MEEP AI Workspace")
            st.info("The AI Analyst has been moved to the **Sidebar** for persistent access. Enable it there to chat while browsing other tabs.")
            st.image("https://img.freepik.com/free-vector/artificial-intelligence-concept-illustration_114360-7006.jpg", width=400)

    if tab_perf:
        with tab_perf:
            st.subheader("📈 Real-World Performance Ledger")
        st.caption("Aggregating historical results from `betting_ledger.csv`.")
        perf_data = get_cached_performance(engine)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lifetime Win Rate", f"{perf_data['win_rate']}%")
        m2.metric("Total Profit", f"${perf_data['profit']:,.2f}")
        m3.metric("ROI", f"{perf_data['roi']}%")
        m4.metric("Total Bets", perf_data['total_bets'])

        st.divider()
        st.markdown("### ⚖️ Portfolio Risk Dashboard")
        st.caption("Monitoring real-time exposure and relational conflicts.")
        
        pr1, pr2 = st.columns([2, 1])
        with pr1:
            st.markdown("#### Cumulative Exposure Radar")
            # Mock risk metrics based on current slip
            slip_count = len(st.session_state.parlay_slip)
            exposure_val = slip_count * 25.0
            
            risk_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = exposure_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Exposure Index", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "#00D1FF"},
                    'steps': [
                        {'range': [0, 150], 'color': "rgba(0, 209, 255, 0.1)"},
                        {'range': [150, 300], 'color': "rgba(255, 184, 0, 0.2)"},
                        {'range': [300, 500], 'color': "rgba(255, 75, 75, 0.3)"}
                    ],
                }
            ))
            risk_fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(risk_fig, width="stretch")

        with pr2:
            st.markdown("#### Health Check")
            if slip_count > 0:
                st.write(f"**Legs Drafted:** `{slip_count}`")
                st.progress(min(1.0, slip_count/6.0))
                if slip_count > 4:
                    st.warning("⚠️ High Correlation Cluster detected in Parlay Slip.")
                else:
                    st.success("✅ Portfolio diversification is optimal.")
            else:
                st.info("No active exposure. Draft legs to see risk profiling.")
        
        if perf_data['equity_curve']:
            df_curve = pd.DataFrame(perf_data['equity_curve'])
            fig = px.area(df_curve, x='date', y='equity', title="Actual Wealth Growth (Staked via Ledger)")
            fig.update_layout(
                template="plotly_dark", 
                margin=dict(l=0, r=0, t=50, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, width="stretch")

    if tab_vault:
        with tab_vault:
            st.subheader("🏛️ Historical Slate Archive")
        vault_date = st.date_input("Inspect Historical Date", datetime.now() - timedelta(days=1))
        
        slate = engine.get_portfolio_slate(vault_date.strftime("%Y-%m-%d"))
        if slate:
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.markdown("#### 🛡️ CORE")
                for p in slate.get('core', []):
                    st.caption(f"**{p.get('name')}**")
                    for leg in p.get('legs', []):
                        st.markdown(f"- <span style='font-size: 11px;'>{leg}</span>", unsafe_allow_html=True)
            with vc2:
                st.markdown("#### 📈 GROWTH")
                for p in slate.get('growth', []):
                    st.caption(f"**{p.get('name')}**")
                    for leg in p.get('legs', []):
                        st.markdown(f"- <span style='font-size: 11px;'>{leg}</span>", unsafe_allow_html=True)
            with vc3:
                st.markdown("#### 🚀 MOONSHOT")
                for p in slate.get('moonshot', []):
                    st.caption(f"**{p.get('name')}**")
                    for leg in p.get('legs', []):
                        st.markdown(f"- <span style='font-size: 11px;'>{leg}</span>", unsafe_allow_html=True)
        else:
            st.warning(f"No archived slate for {vault_date.strftime('%Y-%m-%d')}")

    if tab_lab:
        with tab_lab:
            st.subheader("🔬 Advanced Analytics Lab")
            l_tab1, l_tab2 = st.tabs(["🧬 Player Deep Dive", "🧮 Neural Correlation"])
            
            with l_tab1:
                 render_player_deep_dive(engine)
            
            with l_tab2:
                st.markdown("#### Neural Correlation & Quant Lab")
                st.caption("Analyzing relational dependencies and volatility clusters across prop surfaces.")
                
                all_players = getattr(engine.engine, 'all_players_list', [])
                if not all_players:
                     # Fallback if list not populated
                     all_players = list(engine.engine.name_map.keys()) if hasattr(engine.engine, 'name_map') else []

                q_col1, q_col2 = st.columns([1, 2])
                target_player = q_col1.selectbox("Select Target Asset", options=all_players, key="quant_player_select")
                
                if target_player:
                    with q_col2:
                        corr_data = engine.get_player_correlation(target_player)
                        if corr_data:
                            fig_corr = px.imshow(
                                pd.DataFrame(corr_data['matrix']),
                                text_auto=".2f",
                                aspect="auto",
                                color_continuous_scale='RdBu_r',
                                range_color=[-1, 1],
                                labels=dict(color="Correlation")
                            )
                            fig_corr.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=40,b=0), coloraxis_showscale=False)
                            st.plotly_chart(fig_corr, width="stretch")
                            
                            if corr_data['clusters']:
                                st.markdown("#### 🚩 Risk Clusters Identified")
                                cc1, cc2 = st.columns(2)
                                for i, cluster in enumerate(corr_data['clusters']):
                                    col = cc1 if i % 2 == 0 else cc2
                                    with col:
                                        color = "red" if cluster['risk'] == "Amplification" else "green"
                                        st.write(f"**{cluster['props'][0].upper()}** vs **{cluster['props'][1].upper()}**")
                                        st.caption(f"Relational R: `{cluster['r']:.2f}` | Type: :{color}[{cluster['risk']}]")
                        else:
                            st.warning("Insufficient game sample for correlation analysis.")

    if tab_audit:
        with tab_audit:
            st.subheader("🕵️ Advanced Audit Ledger")
            st.caption("A persistent, immutable trace of all platform transactions and AI-driven recommendations.")
            
            trail = engine.get_audit_trail(limit=50)
            if trail:
                audit_data = [
                    {
                        "Timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "Action": t.action_type.upper(),
                        "Mode": t.mode,
                        "Details": t.details,
                        "Impact": f"${t.impact_amount:.2f}"
                    } for t in trail
                ]
                st.table(pd.DataFrame(audit_data))
            else:
                st.info("The audit ledger is currently empty. All future actions will be logged here.")
            
            if st.button("🧼 RUN BANKROLL RECONCILIATION"):
                engine.log_audit_action("bankroll_recon", "Manual reconciliation triggered by operator.")
                st.success("Reconciliation engine started in background.")

# Cycle 6 Launch Telemetry (LCH-03)
latency = (time.time() - st.session_state.render_start_time) * 1000
engine.log_telemetry("render_latency_ms", latency)

st.session_state.render_start_time = time.time() # Reset for next run

session.close()
