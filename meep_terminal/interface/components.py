import streamlit as st
import pandas as pd
from datetime import datetime
from meep_terminal.data.models import Bet

class ExecutionComponent:
    @staticmethod
    def render_bet_slip(session):
        """Render the Parlay Builder / Single Bet Slip sidebar/panel."""
        st.markdown("### 📋 Bet Slip")
        
        if not st.session_state.parlay_slip:
            st.info("Your slip is empty. Click 'Add' in the table to add legs.")
            return

        total_stake = 0
        
        # Action Bar Top
        col_c1, col_c2 = st.columns([2,1])
        if col_c2.button("🗑️ Clear", width="stretch"):
            st.session_state.parlay_slip = []
            st.rerun()

        st.divider()

        for i, item in enumerate(st.session_state.parlay_slip):
            with st.container():
                c_head1, c_head2 = st.columns([5,1])
                c_head1.markdown(f"**{item['player']}**")
                c_head1.caption(f"`{item['prop'].upper()}` @ {item.get('line',0)}")
                if c_head2.button("✕", key=f"rem_{i}", help="Remove leg"):
                    st.session_state.parlay_slip.pop(i)
                    st.rerun()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Line", value=float(item.get('line', 0.0)), key=f"line_{i}", label_visibility="collapsed", format="%.1f")
                with col2:
                    # Adversarial Fix: Prevent negative or zero stakes
                    stake = st.number_input("Stake $", value=10.0, min_value=0.01, step=1.0, key=f"stake_{i}", label_visibility="collapsed")
                    total_stake += stake
                
                st.divider()

        # Total Exposure Box
        st.metric("Total Exposure", f"${total_stake:.2f}")
        
        # === MEEP HIGH-ALPHA BETTING CARD (Riq's Picks Logic) ===
        if len(st.session_state.parlay_slip) >= 3:
            st.markdown("#### ⚡ Portfolio Optimizer")
            
            import itertools
            import numpy as np
            
            # 1. Sort by Win Prob for Core/Growth
            picks_by_prob = sorted(st.session_state.parlay_slip, key=lambda x: x.get('win_prob', 0), reverse=True)
            # Sort by EV for Moonshot
            picks_by_ev = sorted(st.session_state.parlay_slip, key=lambda x: x.get('ev', 0), reverse=True)
            
            core = picks_by_prob[:3]
            growth = picks_by_prob[3:9]
            
            # Robust mapping for IDs
            core_ids = [str(x.get('id', '')) for x in core]
            moonshot = [p for p in picks_by_ev if str(p.get('id', '')) not in core_ids][:3]
            
            # Use columns for sub-tiers to save space
            sub1, sub2 = st.columns(2)
            
            with sub1:
                with st.expander("🛡️ Core", expanded=True):
                    for p in core:
                        st.caption(f"**{p['player']}** ({p.get('win_prob', 0):.0f}%)")
            
            with sub2:
                 with st.expander("📈 Growth", expanded=False):
                    if len(growth) >= 2:
                        combos = list(itertools.combinations(growth, 2))
                        for combo in combos[:2]:
                            st.caption(f"**{combo[0]['player']}** + **{combo[1]['player']}**")
            
            with st.expander("🚀 Moonshot (Alpha RRs)", expanded=False):
                if len(moonshot) >= 3:
                    for p in moonshot:
                        st.caption(f"**{p['player']}** (Edge: +{p.get('ev',0):.1f}%)")
                st.info("Suggested: **Round Robin (3x2)**")
        
        st.divider()
        
        if st.button("💾 SYNC TO LEDGER", type="primary", width="stretch"):
            try:
                for i, item in enumerate(st.session_state.parlay_slip):
                    new_bet = Bet(
                        pick_id=item['id'],
                        status="placed",
                        stake_amount=st.session_state[f"stake_{i}"],
                        actual_line=st.session_state[f"line_{i}"],
                        actual_odds=-110, # Defaulting for now
                        placed_at=datetime.utcnow()
                    )
                    session.add(new_bet)
                
                session.commit()
                st.session_state.parlay_slip = []
                st.success("Execution logs preserved.")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"Sync failed: {e}")

class AnalyticsComponent:
    @staticmethod
    def render_performance_metrics(placed_bets):
        """Visual summary of ROI, hit rate, and streaks."""
        if not placed_bets:
            st.info("No execution history found.")
            return

        df = pd.DataFrame(placed_bets)
        st.metric("Total Bets", len(df))
