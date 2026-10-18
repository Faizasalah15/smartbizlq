"""Configuration module for SmartBizLQ.

Loads environment variables from .env and exposes configuration settings,
application constants, and database URL resolution logic.
"""

import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Application Constants
APP_NAME: str = "SmartBizLQ"
APP_VERSION: str = "1.1.0"

# Database Configuration
DB_USER: str | None = os.getenv("DB_USER")
DB_PASS: str | None = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD")
DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT: str = os.getenv("DB_PORT", "3306")
DB_NAME: str | None = os.getenv("DB_NAME")
SQLITE_FILE: str = os.getenv("SQLITE_FILE", "smartbizlq_local.db")

# SMTP Configuration (for OTP password reset emails)
SMTP_HOST: str | None = os.getenv("SMTP_HOST")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587")) if os.getenv("SMTP_PORT") else 587
SMTP_USER: str | None = os.getenv("SMTP_USER")
SMTP_PASS: str | None = os.getenv("SMTP_PASS") or os.getenv("SMTP_PASSWORD")

# Optional: the account registered with this email is given the Admin role.
ADMIN_EMAIL: str | None = (os.getenv("ADMIN_EMAIL") or "").strip().lower() or None

# Development Mode Flag
DEV_MODE: bool = os.getenv("DEV_MODE", "false").strip().lower() in ("true", "1", "t", "yes")


def get_database_url() -> str:
    """Return the database connection URL based on environment settings.

    Attempts to construct a MySQL connection URL if DB_USER, DB_PASS, and DB_NAME
    are present. Otherwise, falls back to a local SQLite database URL.

    Returns:
        str: SQLAlchemy-compatible database connection URL.
    """
    if DB_USER and DB_PASS and DB_NAME:
        port_suffix = f":{DB_PORT}" if DB_PORT else ""
        url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}{port_suffix}/{DB_NAME}"
        logger.info("Configured MySQL connection for host '%s', database '%s'.", DB_HOST, DB_NAME)
        return url

    sqlite_url = f"sqlite:///{SQLITE_FILE}"
    logger.info("Using local SQLite database URL: %s", sqlite_url)
    return sqlite_url
