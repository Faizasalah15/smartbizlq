"""Unit tests for smartbizlq.auth modules (utils, signup, login, password_reset)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from smartbizlq.auth.login import authenticate_user
from smartbizlq.auth.password_reset import create_and_store_otp, validate_otp
from smartbizlq.auth.signup import create_user
from smartbizlq.auth.utils import hash_password, verify_password
from smartbizlq.db.models import Base, User


@pytest.fixture
def test_db_session(monkeypatch):
    """Provide an isolated SQLite database session for auth tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    import smartbizlq.auth.login as login_mod
    import smartbizlq.auth.password_reset as reset_mod
    import smartbizlq.auth.signup as signup_mod

    monkeypatch.setattr(signup_mod, "SessionLocal", Session)
    monkeypatch.setattr(login_mod, "SessionLocal", Session)
    monkeypatch.setattr(reset_mod, "SessionLocal", Session)

    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_hash_and_verify_password():
    """Verify password hashing produces valid bcrypt hashes and verifies correctly."""
    pw = "SuperSecret123!"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_create_user_success(test_db_session):
    """Test successful user registration with allowed roles."""
    ok, msg = create_user("test@example.com", "mypassword1", "Test User", role="Viewer")
    assert ok is True
    assert msg == "User created"

    # Verify duplicate email prevention
    ok_dup, msg_dup = create_user("test@example.com", "anotherpass", "Duplicate", role="Analyst")
    assert ok_dup is False
    assert msg_dup == "Email exists"


def test_create_user_invalid_role(test_db_session):
    """Test user registration rejects roles other than Viewer or Analyst."""
    ok, msg = create_user("admin@example.com", "pw", "Admin Guy", role="Admin")
    assert ok is False
    assert "Invalid role" in msg


def test_authenticate_user(test_db_session):
    """Test user login authentication."""
    create_user("auth_test@example.com", "correct_pass", "Auth User", role="Analyst")

    # Correct credentials
    ok, user = authenticate_user("auth_test@example.com", "correct_pass")
    assert ok is True
    assert user.email == "auth_test@example.com"
    assert user.role == "Analyst"

    # Wrong password
    ok_wrong_pw, err_msg = authenticate_user("auth_test@example.com", "wrong_pass")
    assert ok_wrong_pw is False
    assert err_msg == "Invalid password"

    # Nonexistent user
    ok_nonexistent, err_msg2 = authenticate_user("nonexistent@example.com", "pass")
    assert ok_nonexistent is False
    assert err_msg2 == "User not found"


def test_create_and_validate_otp(test_db_session):
    """Test OTP creation and successful validation."""
    email = "reset@example.com"
    token = create_and_store_otp(email, test_db_session, expiry_minutes=15)
    assert len(token) == 6
    assert token.isdigit()

    # Valid token verification
    ok, msg = validate_otp(email, token, test_db_session)
    assert ok is True
    assert msg == "OK"

    # Re-using the same OTP should fail (used=True)
    ok_reuse, msg_reuse = validate_otp(email, token, test_db_session)
    assert ok_reuse is False
    assert msg_reuse == "Invalid token"


def test_validate_otp_expired(test_db_session):
    """Test that expired OTPs are rejected."""
    email = "expired@example.com"
    # Create an already-expired token directly
    token = create_and_store_otp(email, test_db_session, expiry_minutes=-5)

    ok, msg = validate_otp(email, token, test_db_session)
    assert ok is False
    assert msg == "Token expired"


def test_signup_cannot_self_assign_admin_but_admin_email_can(test_db_session, monkeypatch):
    import smartbizlq.auth.signup as signup

    ok, msg = signup.create_user("someone@example.com", "pw123456", "Someone", role="Admin")
    assert not ok and "Invalid role" in msg

    monkeypatch.setattr(signup, "ADMIN_EMAIL", "boss@example.com")
    ok, _ = signup.create_user("Boss@Example.com", "pw123456", "Boss", role="Viewer")
    assert ok
    assert test_db_session.query(User).filter(User.email == "boss@example.com").one().role == "Admin"
