"""Authentication utilities for password hashing and verification."""

import logging
import bcrypt

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt.

    Args:
        password: The plaintext password string to hash.

    Returns:
        str: Salted and hashed password as a UTF-8 string.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a candidate plaintext password against a stored bcrypt hash.

    Args:
        password: The plaintext candidate password.
        hashed: The stored bcrypt hash string.

    Returns:
        bool: True if password matches the hash, False otherwise.
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as e:
        logger.warning("Password verification encountered an error: %s", e)
        return False
