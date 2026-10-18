"""User registration and sign-up logic for SmartBizLQ.

Handles user creation with role restrictions and provides the Streamlit
sign-up interface.
"""

import logging
import streamlit as st

from ..config import ADMIN_EMAIL
from ..db.models import User
from ..db.session import SessionLocal
from .utils import hash_password

logger = logging.getLogger(__name__)

ALLOWED_ROLES: tuple[str, ...] = ("Viewer", "Analyst")


def create_user(
    email: str,
    pw: str,
    name: str,
    role: str = "Viewer",
    phone: str | None = None,
) -> tuple[bool, str]:
    """Create a new user account in the database.

    Args:
        email: User email address.
        pw: Plaintext password.
        name: User's full name.
        role: User role, restricted to 'Viewer' or 'Analyst' (defaults to 'Viewer').
        phone: Optional phone number.

    Returns:
        tuple[bool, str]: (success_flag, status_message).
    """
    if not email or not pw:
        return False, "Email and password are required"

    if role not in ALLOWED_ROLES:
        return False, f"Invalid role: {role}. Allowed roles are: {', '.join(ALLOWED_ROLES)}"

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email.strip().lower()).first()
        if existing:
            logger.info("Registration failed: email '%s' already exists.", email)
            return False, "Email exists"

        clean_email = email.strip().lower()
        # Public sign-up can only choose Viewer or Analyst. The single admin account is
        # configured by setting ADMIN_EMAIL in .env before that person registers.
        if ADMIN_EMAIL and clean_email == ADMIN_EMAIL:
            role = "Admin"
        user = User(
            email=clean_email,
            password_hash=hash_password(pw),
            full_name=name.strip() if name else None,
            role=role,
            phone=phone.strip() if phone else None,
        )
        db.add(user)
        db.commit()
        logger.info("Created user '%s' with role '%s'.", email, role)
        return True, "User created"
    except Exception as e:
        db.rollback()
        logger.error("Database error while creating user '%s': %s", email, e, exc_info=True)
        return False, f"DB error: {e}"
    finally:
        db.close()


def render_signup_page() -> None:
    """Render the Streamlit sign-up UI component.

    Users can register accounts with either 'Viewer' or 'Analyst' role.
    """
    st.header("Sign up")
    su_email = st.text_input("Email", key="su_email")
    su_name = st.text_input("Full name", key="su_name")
    su_phone = st.text_input("Phone (optional)", key="su_phone")
    su_pw = st.text_input("Password", type="password", key="su_pw")
    su_role = st.selectbox("Role", list(ALLOWED_ROLES), key="su_role")

    if st.button("Create account", key="su_create"):
        if not su_email or not su_pw:
            st.error("Please enter both email and password.")
            return

        ok, msg = create_user(su_email, su_pw, su_name, su_role, su_phone)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
