"""Database engine and session management for SmartBizLQ.

Provides resilient database connection handling with MySQL primary support
and automatic SQLite fallback.
"""

import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ..config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER, SQLITE_FILE
from .models import Base

logger = logging.getLogger(__name__)

USING_SQLITE_FALLBACK: bool = True


def create_app_engine() -> Engine:
    """Create SQLAlchemy database engine, attempting MySQL first with SQLite fallback.

    If MySQL configuration is provided (DB_USER, DB_PASS, DB_NAME), attempts to
    connect and ping the MySQL server. If the connection fails or credentials
    are absent, falls back to a local SQLite database.

    Returns:
        Engine: Configured SQLAlchemy database engine.
    """
    global USING_SQLITE_FALLBACK

    if DB_USER and DB_PASS and DB_NAME:
        port_suffix = f":{DB_PORT}" if DB_PORT else ""
        mysql_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}{port_suffix}/{DB_NAME}"
        try:
            eng = create_engine(mysql_url, echo=False, pool_pre_ping=True)
            conn = eng.connect()
            conn.close()
            USING_SQLITE_FALLBACK = False
            logger.info("Connected successfully to MySQL database '%s'.", DB_NAME)
            return eng
        except Exception as e:
            logger.warning(
                "Could not connect to MySQL using provided credentials. Falling back to SQLite. Error: %s",
                e,
            )

    sqlite_url = f"sqlite:///{SQLITE_FILE}"
    eng = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})
    USING_SQLITE_FALLBACK = True
    logger.info("Initialized SQLite database engine at '%s'.", sqlite_url)
    return eng


# Module-level engine and sessionmaker
engine: Engine = create_app_engine()
SessionLocal: sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Columns added after the first release, with their SQL type. Databases created by the
# original app are upgraded in place so existing users and history are kept.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "model_runs": {"dataset_name": "VARCHAR(255)"},
}


def upgrade_schema(eng: Engine) -> list[str]:
    """Add any columns missing from tables created by an older version.

    Returns:
        The ``table.column`` names that were added.
    """
    added: list[str] = []
    insp = inspect(eng)
    existing_tables = set(insp.get_table_names())
    for table, cols in _ADDED_COLUMNS.items():
        if table not in existing_tables:
            continue
        present = {c["name"] for c in insp.get_columns(table)}
        for col, sql_type in cols.items():
            if col not in present:
                with eng.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}"))
                added.append(f"{table}.{col}")
                logger.info("Upgraded database: added column %s.%s", table, col)
    return added


def init_db(target_engine: Engine | None = None) -> None:
    """Initialize database tables for all declared models.

    Args:
        target_engine: Optional engine to bind. Defaults to the module-level engine.
    """
    global engine, SessionLocal, USING_SQLITE_FALLBACK
    eng = target_engine or engine
    try:
        Base.metadata.create_all(bind=eng)
        upgrade_schema(eng)
        logger.info("Database tables verified/created successfully.")
    except Exception as e:
        logger.warning(
            "DB create_all failed on target engine; falling back to SQLite. Error: %s",
            e,
        )
        sqlite_url = f"sqlite:///{SQLITE_FILE}"
        engine = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        USING_SQLITE_FALLBACK = True
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created using SQLite fallback.")


# Initialize tables upon module import
init_db()
