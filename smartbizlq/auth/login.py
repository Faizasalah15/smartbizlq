"""Authentication and login routines for SmartBizLQ.

Provides credential verification and the Streamlit login interface.
"""

import logging
from typing import Any
import streamlit as st

from ..db.models import User
from ..db.session import SessionLocal
from .utils import verify_password

logger = logging.getLogger(__name__)


def authenticate_user(email: str, pw: str) -> tuple[bool, Any]:
    """Authenticate a user using their email and plaintext password.

    Args:
        email: User email address.
        pw: Candidate plaintext password.

    Returns:
        tuple[bool, Any]: (True, User_object) on success, or
            (False, error_message_string) on failure.
    """
    if not email or not pw:
        return False, "Email and password are required"

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email.strip().lower()).first()
        if not user:
            logger.info("Authentication failed: user '%s' not found.", email)
            return False, "User not found"

        if verify_password(pw, user.password_hash):
            logger.info("Authentication successful for user '%s'.", email)
            return True, user

        logger.warning("Authentication failed: invalid password for user '%s'.", email)
        return False, "Invalid password"
    except Exception as e:
        logger.error("Error during authentication for user '%s': %s", email, e, exc_info=True)
        return False, f"Authentication error: {e}"
    finally:
        db.close()


def render_login_page() -> None:
    """Render the Streamlit login UI.

    Upon successful authentication, updates `st.session_state.user` with
    the authenticated user profile dictionary.
    """
    st.header("Login")
    li_email = st.text_input("Email", key="li_email")
    li_pw = st.text_input("Password", type="password", key="li_pw")

    if st.button("Login", key="li_login"):
        if not li_email or not li_pw:
            st.error("Please enter email and password.")
            return

        ok, res = authenticate_user(li_email, li_pw)
        if ok:
            u = res
            st.session_state.user = {
                "id": u.id,
                "email": u.email,
                "role": u.role,
                "name": u.full_name,
            }
            st.success("Logged in")
            st.rerun()
        else:
            st.error(res)
