"""Home landing page with user score, achievement badges, and platform navigation overview."""

import logging
from datetime import timezone

import streamlit as st

logger = logging.getLogger(__name__)

from ..db.models import User, UserBadge
from ..db.session import SessionLocal


def render_home_page() -> None:
    """Render the Home landing page with user profile score, gamification badges, and platform overview."""
    st.header("SmartBizLQ — Business Analytics, Forecasting & Scenario Planning")

    user_info = st.session_state.get("user", {})
    user_name = user_info.get("name", "User")
    user_role = user_info.get("role", "Viewer")
    user_id = user_info.get("id")

    st.markdown(f"Welcome back, **{user_name}** (`{user_role}`).")
    st.markdown(
        "SmartBizLQ provides end-to-end data analytics: upload datasets, build dynamic Power BI-style "
        "visualizations, train machine learning models, run financial scenario models, and export presentation-ready reports."
    )

    # -------------------------------------------------------------
    # Gamification: Score & Badges
    # -------------------------------------------------------------
    score_val = 0
    badges_list = []

    if SessionLocal is not None and user_id is not None:
        try:
            with SessionLocal() as db:
                if User is not None:
                    u = db.query(User).get(user_id)
                    if u and u.score:
                        score_val = u.score
                if UserBadge is not None:
                    badges = db.query(UserBadge).filter(UserBadge.user_id == user_id).all()
                    badges_list = [
                        (b.badge, b.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d") if b.created_at else "Recent")
                        for b in badges
                    ]
        except Exception as e:
            logger.warning("Could not load user profile gamification stats: %s", e)

    col_score, col_badges = st.columns([1, 2])

    with col_score:
        st.subheader("Analyst Score")
        st.metric("Engagement Points", f"{score_val:,}", help="Earn points by uploading datasets, running ML models, and generating reports.")

    with col_badges:
        st.subheader("Earned Badges")
        if not badges_list:
            st.info("No badges earned yet. Train a model in ML Studio or export a report to unlock badges!")
        else:
            for badge_name, badge_date in badges_list:
                st.markdown(f"- 🎖️ **{badge_name}** *(Earned {badge_date})*")

    st.markdown("---")

    # -------------------------------------------------------------
    # Platform Capabilities Guide
    # -------------------------------------------------------------
    st.subheader("Platform Capabilities")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("### 📤 Upload & Clean")
        st.write(
            "Ingest raw CSV files with automatic delimiter sniffing, missing-value profiling, "
            "duplicate pruning, and session staging."
        )

        st.markdown("### 📊 Executive Dashboard")
        st.write(
            "Cross-filter KPIs, revenue velocity, top categories, and geographic choropleths with "
            "one-click PowerPoint and Excel workbook exports."
        )

    with g2:
        st.markdown("### 📈 Visuals Hub")
        st.write(
            "Explore 24 Power BI-style visuals including 100% stacked bars, combo charts, "
            "waterfalls, funnels, pivot matrices, and heatmaps."
        )

        st.markdown("### 🤖 ML Studio")
        st.write(
            "Train Prophet forecasts with prediction intervals, customer churn classifiers "
            "with ROC-AUC and feature importance, and KMeans segmentations with elbow curves."
        )

    with g3:
        st.markdown("### 💼 Financial Scenarios")
        st.write(
            "Simulate Price, Volume, Cost, and Growth rate fluctuations with tornado sensitivity "
            "charts and export workbooks containing live Excel formulas."
        )

        st.markdown("### 📜 Audit & Admin")
        st.write(
            "Audit historical model runs with full hyperparameter tracking, manage user roles, "
            "and inspect system telemetry."
        )
