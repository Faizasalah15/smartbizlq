"""Unit tests for smartbizlq.config module."""

from smartbizlq.config import (
    APP_NAME,
    APP_VERSION,
    DEV_MODE,
)


def test_app_constants():
    """Verify application name and version constants."""
    assert APP_NAME == "SmartBizLQ"
    assert isinstance(APP_VERSION, str)
    assert len(APP_VERSION) > 0


def test_dev_mode_type():
    """Verify DEV_MODE is a boolean."""
    assert isinstance(DEV_MODE, bool)


def test_get_database_url_fallback(monkeypatch):
    """Verify get_database_url falls back to SQLite when MySQL env vars are absent."""
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASS", raising=False)
    monkeypatch.delenv("DB_NAME", raising=False)

    import smartbizlq.config as cfg
    monkeypatch.setattr(cfg, "DB_USER", None)
    monkeypatch.setattr(cfg, "DB_PASS", None)
    monkeypatch.setattr(cfg, "DB_NAME", None)

    url = cfg.get_database_url()
    assert url.startswith("sqlite:///")
    assert url == f"sqlite:///{cfg.SQLITE_FILE}"


def test_get_database_url_mysql(monkeypatch):
    """Verify get_database_url constructs valid MySQL URL when env vars are present."""
    import smartbizlq.config as cfg
    monkeypatch.setattr(cfg, "DB_USER", "test_user")
    monkeypatch.setattr(cfg, "DB_PASS", "test_pass")
    monkeypatch.setattr(cfg, "DB_HOST", "127.0.0.1")
    monkeypatch.setattr(cfg, "DB_PORT", "3306")
    monkeypatch.setattr(cfg, "DB_NAME", "test_db")

    url = cfg.get_database_url()
    assert url == "mysql+pymysql://test_user:test_pass@127.0.0.1:3306/test_db"
