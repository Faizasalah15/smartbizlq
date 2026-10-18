"""SQLAlchemy ORM models for SmartBizLQ.

Defines database schemas for users, datasets, model runs, OTP resets,
and gamification badges.
"""

from datetime import datetime, timezone
from typing import Any
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base: Any = declarative_base()


class User(Base):
    """User account model for authentication, role-based access, and scoring."""

    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True)
    email: str = Column(String(255), unique=True, nullable=False)
    password_hash: str = Column(String(255), nullable=False)
    full_name: str | None = Column(String(255), nullable=True)
    role: str = Column(String(50), default="Viewer")
    phone: str | None = Column(String(50), nullable=True)
    score: int = Column(Integer, default=0)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"


class Dataset(Base):
    """Metadata record for uploaded and cleaned datasets."""

    __tablename__ = "datasets"

    id: int = Column(Integer, primary_key=True)
    filename: str = Column(String(255), nullable=False)
    schema_json: str | None = Column(Text, nullable=True)
    rows_count: int | None = Column(Integer, nullable=True)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, filename='{self.filename}', rows={self.rows_count})>"


class ModelRun(Base):
    """Record of a trained machine learning model run, metrics, and parameters."""

    __tablename__ = "model_runs"

    id: int = Column(Integer, primary_key=True)
    model_name: str = Column(String(255), nullable=False)
    version: str | None = Column(String(100), nullable=True)
    metrics_json: str | None = Column(Text, nullable=True)
    params_json: str | None = Column(Text, nullable=True)
    file_path: str | None = Column(String(512), nullable=True)
    dataset_name: str | None = Column(String(255), nullable=True)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (
            f"<ModelRun(id={self.id}, model_name='{self.model_name}', "
            f"version='{self.version}', dataset='{self.dataset_name}')>"
        )


class OTPReset(Base):
    """One-Time Password token for user password reset requests."""

    __tablename__ = "otp_resets"

    id: int = Column(Integer, primary_key=True)
    email: str = Column(String(255), index=True, nullable=False)
    token_hash: str = Column(String(255), nullable=False)
    expires_at: datetime = Column(DateTime, nullable=False)
    used: bool = Column(Boolean, default=False)

    def __repr__(self) -> str:
        return f"<OTPReset(id={self.id}, email='{self.email}', used={self.used})>"


class UserBadge(Base):
    """Gamification badge awarded to a user for platform activities."""

    __tablename__ = "user_badges"

    id: int = Column(Integer, primary_key=True)
    user_id: int = Column(Integer, index=True, nullable=False)
    badge: str = Column(String(255), nullable=False)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<UserBadge(id={self.id}, user_id={self.user_id}, badge='{self.badge}')>"
