"""SmartBizLQ application shell: page setup, sign-in gate and navigation.

Run from the project root with:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from . import __version__
from .auth.login import render_login_page
from .auth.password_reset import render_password_reset_page
from .auth.signup import render_signup_page
from .ui.admin import render_admin_page
from .ui.dashboard import render_dashboard_page
from .ui.home import render_home_page
from .ui.ml_studio import render_ml_studio_page
from .ui.model_history import render_model_history_page
from .ui.scenarios import render_scenarios_page
from .ui.upload import render_upload_page
from .ui.visuals_hub import render_visuals_hub_page

GUEST_USER = {"id": None, "email": None, "name": "Guest", "role": "Viewer"}

PAGES = {
    "Home": render_home_page,
    "Upload data": render_upload_page,
    "Dashboard": render_dashboard_page,
    "Visuals Hub": render_visuals_hub_page,
    "ML Studio": render_ml_studio_page,
    "Scenario Studio": render_scenarios_page,
    "Model history": render_model_history_page,
}


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _render_sign_in() -> None:
    st.title("SmartBizLQ")
    st.caption("Self-service business analytics: dashboards, forecasting, churn modelling and scenario planning.")
    if st.button("Explore as guest (demo data only)", key="guest_login", type="primary"):
        st.session_state.user = dict(GUEST_USER)
        st.rerun()
    tab_login, tab_signup, tab_reset = st.tabs(["Log in", "Sign up", "Reset password"])
    with tab_login:
        render_login_page()
    with tab_signup:
        render_signup_page()
    with tab_reset:
        render_password_reset_page()


def main() -> None:
    """Entry point called by ``streamlit_app.py``."""
    st.set_page_config(page_title="SmartBizLQ", layout="wide")
    _configure_logging()

    user = st.session_state.get("user")
    if not user:
        _render_sign_in()
        return

    pages = dict(PAGES)
    if user.get("role") == "Admin":
        pages["Admin"] = render_admin_page

    with st.sidebar:
        st.markdown(f"### SmartBizLQ `v{__version__}`")
        st.caption(f"Signed in as **{user.get('name') or user.get('email')}** ({user.get('role')})")
        choice = st.radio("Go to", list(pages), key="nav_page")
        if st.button("Log out", key="logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.divider()

    pages[choice]()
