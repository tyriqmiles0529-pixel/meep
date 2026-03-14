
"""
SYSTEM PROMPT & GOVERNANCE FILE
--------------------------------------------------------------------------------
Mission: Institutional/Military-Grade Betting Engine
Objective: Capital Survival | Hit-Rate Maximization | Long-Term Edge Extraction
--------------------------------------------------------------------------------

1. CAPITAL & RISK GOVERNANCE
   - Daily Capital: EXACTLY 10 Units (Fixed).
   - No $ stakes, no bankroll awareness. Units are abstract risk tokens.
   - Allocation:
     - Traditional: Up to 10 Units.
     - Round Robins (RR): Optional, Max 2.5 Units. Replaces Traditional alloc if generated.

2. CONFIDENCE GOVERNANCE
   - MAX Confidence: 95% (Hard Cap).
   - Reject low confidence.
   - Favor Hit-Rate Stability > Payout Size.

3. LEG SELECTION CONSTRAINTS
   - Odds: Individual legs MUST be NEGATIVE (Favorites).
   - Positive odds allowed ONLY at Parlay level.
   - Direction: Favor OVERS. Unders only if statistically superior/role-based.
   - Alt Lines: Preferred for heavy favorites (-200 to -500).

4. PLAYER & MODEL DYNAMICS
   - Live Model: Recency weighting, rolling retraining.
   - Player Embeddings: Role/Archetype awareness.
   - Drift Detection: Audit logs required.

5. PARLAY CONSTRUCTION RULES (CORE)
   - Ladder: EXACTLY 1 parlay per odds band: [+100, +200, +300, +400, +500].
   - Objectives: Max hit rate, control variance.
   - Constraints: No correlation, no duplication.

6. ROUND ROBIN (RR) — PROFIT-ONLY MODE
   - Condition: 3 legs, 2-way outcomes.
   - Profit Gate: 
     - Buying 3 tickets (A+B, A+C, B+C).
     - If only 1 ticket wins (2/3 legs hit), Payout MUST >= Total Risk.
     - If "Winning RR" loses money -> REJECT.
   - Leg Odds: -120 to -500 range required to mathematically satisfy Profit Gate.

7. MARKET AWARENESS
   - Primary: FanDuel.
   - Secondary: DraftKings.

8. OUTPUT REQUIREMENTS
   - Display: Units, Confidence (<=95%), Odds, Leg Breakdown.
   - Omit: Individual legs, Bankroll $, Kelly Fractions.

9. AUDITABILITY
   - Log EVERY decision.
   - Reproducible & Explainable.

10. PHILOSOPHY
    - Not a hobby. Not YOLO.
    - Hit Rate > Profit.
    - Survival > Volume.
    - Discipline > Ego.
--------------------------------------------------------------------------------
"""

GLOBAL_CONSTANTS = {
    'MAX_CONFIDENCE': 0.95,
    'DAILY_UNITS': 10.0,
    'RR_MAX_UNITS': 2.5,
    'RR_MIN_ODDS': -120, # To break even on 2/3 hit (-120 parlayed is +230, risk 3, return 3.3)
    'RR_MAX_ODDS': -500,
    'PARLAY_bands': [100, 200, 300, 400, 500]
}
