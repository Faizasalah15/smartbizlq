"""Unit tests for smartbizlq.db module (models and session)."""

from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from smartbizlq.db.models import (
    Base,
    Dataset,
    ModelRun,
    OTPReset,
    User,
    UserBadge,
)
from smartbizlq.db.session import create_app_engine, init_db


@pytest.fixture
def in_memory_db():
    """Create a temporary in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_user_model(in_memory_db):
    """Test User ORM model insertion and representation."""
    user = User(
        email="analyst@smartbizlq.com",
        password_hash="fake_hash",
        full_name="Jane Doe",
        role="Analyst",
        phone="555-0100",
        score=10,
    )
    in_memory_db.add(user)
    in_memory_db.commit()

    retrieved = in_memory_db.query(User).filter_by(email="analyst@smartbizlq.com").first()
    assert retrieved is not None
    assert retrieved.full_name == "Jane Doe"
    assert retrieved.role == "Analyst"
    assert "<User" in repr(retrieved)


def test_dataset_model(in_memory_db):
    """Test Dataset ORM model insertion."""
    dataset = Dataset(
        filename="sales_2025.csv",
        schema_json='{"sales": "float64"}',
        rows_count=500,
    )
    in_memory_db.add(dataset)
    in_memory_db.commit()

    retrieved = in_memory_db.query(Dataset).filter_by(filename="sales_2025.csv").first()
    assert retrieved is not None
    assert retrieved.rows_count == 500
    assert "<Dataset" in repr(retrieved)


def test_model_run_model(in_memory_db):
    """Test ModelRun ORM model with the new dataset_name field."""
    run = ModelRun(
        model_name="ProphetForecaster",
        version="1.0.0",
        metrics_json='{"mae": 12.5}',
        params_json='{"changepoint_prior_scale": 0.05}',
        file_path="models/test_model.joblib",
        dataset_name="sales_2025.csv",
    )
    in_memory_db.add(run)
    in_memory_db.commit()

    retrieved = in_memory_db.query(ModelRun).filter_by(model_name="ProphetForecaster").first()
    assert retrieved is not None
    assert retrieved.dataset_name == "sales_2025.csv"
    assert "<ModelRun" in repr(retrieved)


def test_otp_reset_model(in_memory_db):
    """Test OTPReset ORM model."""
    otp = OTPReset(
        email="user@smartbizlq.com",
        token_hash="fake_sha256",
        expires_at=datetime.now(timezone.utc),
        used=False,
    )
    in_memory_db.add(otp)
    in_memory_db.commit()

    retrieved = in_memory_db.query(OTPReset).filter_by(email="user@smartbizlq.com").first()
    assert retrieved is not None
    assert retrieved.used is False
    assert "<OTPReset" in repr(retrieved)


def test_user_badge_model(in_memory_db):
    """Test UserBadge ORM model."""
    badge = UserBadge(user_id=1, badge="Data Explorer")
    in_memory_db.add(badge)
    in_memory_db.commit()

    retrieved = in_memory_db.query(UserBadge).filter_by(user_id=1).first()
    assert retrieved is not None
    assert retrieved.badge == "Data Explorer"
    assert "<UserBadge" in repr(retrieved)


def test_create_app_engine_and_init_db():
    """Verify create_app_engine returns valid engine and init_db runs successfully."""
    engine = create_app_engine()
    assert engine is not None
    init_db(engine)


def test_upgrade_schema_adds_missing_column_to_old_database(tmp_path):
    """A database created by the original app (no dataset_name column) is upgraded in place."""
    from sqlalchemy import create_engine, inspect, text

    from smartbizlq.db.session import init_db

    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE model_runs (id INTEGER PRIMARY KEY, model_name VARCHAR(255) NOT NULL, version VARCHAR(100), "
            "metrics_json TEXT, params_json TEXT, file_path VARCHAR(512), created_at DATETIME)"
        ))
        conn.execute(text("INSERT INTO model_runs (model_name, version) VALUES ('prophet', 'v1')"))
    init_db(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("model_runs")}
    assert "dataset_name" in cols
    with eng.connect() as conn:
        assert conn.execute(text("SELECT model_name FROM model_runs")).scalar() == "prophet"
