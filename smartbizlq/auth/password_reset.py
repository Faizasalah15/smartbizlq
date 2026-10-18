"""Password reset workflows via One-Time Password (OTP) for SmartBizLQ.

Includes OTP generation, hashing, database storage, email transmission,
and the Streamlit reset UI component.
"""

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import hashlib
import logging
import secrets
import smtplib
from sqlalchemy.orm import Session
import streamlit as st

from ..config import DEV_MODE, SMTP_HOST, SMTP_PASS, SMTP_PORT, SMTP_USER
from ..db.models import OTPReset, User
from ..db.session import SessionLocal
from .utils import hash_password

logger = logging.getLogger(__name__)


def create_and_store_otp(email: str, db_session: Session, expiry_minutes: int = 15) -> str:
    """Generate a random 6-digit OTP, store its SHA-256 hash in the database, and return token.

    Args:
        email: User email address requesting the reset.
        db_session: SQLAlchemy active database session.
        expiry_minutes: Lifespan of the generated OTP in minutes (default 15).

    Returns:
        str: 6-digit plaintext OTP token.
    """
    token: str = str(secrets.randbelow(10**6)).zfill(6)
    token_hash: str = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at: datetime = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)

    rec = OTPReset(
        email=email.strip().lower(),
        token_hash=token_hash,
        expires_at=expires_at,
        used=False,
    )
    db_session.add(rec)
    db_session.commit()
    logger.info("Generated OTP for %s expiring in %d minutes.", email, expiry_minutes)
    return token


def validate_otp(email: str, token: str, db_session: Session) -> tuple[bool, str]:
    """Validate a user-submitted OTP against stored unexpired tokens.

    Args:
        email: User email address.
        token: Plaintext OTP token submitted by user.
        db_session: SQLAlchemy active database session.

    Returns:
        tuple[bool, str]: (True, 'OK') if valid, or (False, error_message) on failure.
    """
    token_hash: str = hashlib.sha256(token.strip().encode("utf-8")).hexdigest()
    rec = (
        db_session.query(OTPReset)
        .filter(
            OTPReset.email == email.strip().lower(),
            OTPReset.token_hash == token_hash,
            OTPReset.used.is_(False),
        )
        .order_by(OTPReset.expires_at.desc())
        .first()
    )

    if not rec:
        logger.warning("OTP validation failed for %s: invalid token.", email)
        return False, "Invalid token"

    rec_expires = rec.expires_at
    if rec_expires.tzinfo is None:
        rec_expires = rec_expires.replace(tzinfo=timezone.utc)

    if rec_expires < datetime.now(timezone.utc):
        logger.warning("OTP validation failed for %s: token expired.", email)
        return False, "Token expired"

    rec.used = True
    db_session.commit()
    logger.info("OTP validated successfully for %s.", email)
    return True, "OK"


def send_otp_email(recipient: str, token: str) -> None:
    """Send OTP token to recipient email via SMTP.

    Args:
        recipient: Destination email address.
        token: Plaintext 6-digit OTP code.

    Raises:
        RuntimeError: If SMTP configuration is missing or incomplete.
    """
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        raise RuntimeError("SMTP not configured in .env")

    msg = EmailMessage()
    msg["Subject"] = "SmartBizLQ password reset code"
    msg["From"] = SMTP_USER
    msg["To"] = recipient
    msg.set_content(
        f"Your SmartBizLQ password reset code is: {token}\nIt expires in 15 minutes."
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    logger.info("Sent OTP reset email to %s.", recipient)


def render_password_reset_page() -> None:
    """Render the Streamlit password reset UI with OTP verification.

    If SMTP delivery fails and DEV_MODE is active, displays the OTP on screen
    for development convenience.
    """
    st.header("Reset password (OTP)")
    rp_email = st.text_input("Registered email", key="rp_email")

    if st.button("Send OTP", key="rp_send"):
        if not rp_email:
            st.error("Please enter your registered email address.")
        else:
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.email == rp_email.strip().lower()).first()
                if not user:
                    st.error("Email not found")
                else:
                    token = create_and_store_otp(rp_email.strip().lower(), db)
                    try:
                        send_otp_email(rp_email.strip().lower(), token)
                        st.success("OTP sent to email")
                    except Exception as ex:
                        logger.warning("SMTP failed for %s: %s", rp_email, ex)
                        st.warning(f"SMTP failed: {ex}")
                        if DEV_MODE:
                            st.info(f"Development OTP (local use): {token}")
                        else:
                            st.info("OTP could not be sent. Check SMTP settings in .env.")
            finally:
                db.close()

    rp_token = st.text_input("OTP", key="rp_token")
    rp_new = st.text_input("New password", type="password", key="rp_new")

    if st.button("Reset password", key="rp_reset"):
        if not rp_email or not rp_token or not rp_new:
            st.error("Provide email, OTP and new password")
        else:
            db = SessionLocal()
            try:
                ok, msg = validate_otp(rp_email.strip().lower(), rp_token.strip(), db)
                if not ok:
                    st.error(msg)
                else:
                    user = db.query(User).filter(User.email == rp_email.strip().lower()).first()
                    if not user:
                        st.error("User not found (unexpected)")
                    else:
                        user.password_hash = hash_password(rp_new)
                        db.commit()
                        st.success("Password reset successful")
            finally:
                db.close()
