# SmartBizLQ_streamlit_app.py
"""
SmartBizLQ_streamlit_app.py
Single-file Streamlit app:
- DB: attempts MySQL (from .env) and falls back to local SQLite automatically
- Upload CSV with robust cleaning (drop empty rows/cols, drop duplicates)
- Dashboard, Visuals Hub, ML Studio with memory-safe model training
- Dynamic controls (selectboxes/multiselects) update based on current dataset
- Unique widget keys to avoid Streamlit duplicate element ID errors
Run:
    pip install -r requirements.txt
    streamlit run SmartBizLQ_streamlit_app.py
"""
import os
import io
import csv
import json
import uuid
import math
import hashlib
import secrets
import smtplib
import re
import traceback
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
import joblib
import bcrypt
from dotenv import load_dotenv
load_dotenv()

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import openai
except Exception:
    openai = None

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance

# pptx
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
except Exception:
    Presentation = None

# -----------------------------
# CONFIG (from .env)
# -----------------------------
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_NAME = os.getenv('DB_NAME')

SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587')) if os.getenv('SMTP_PORT') else 587
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

USING_SQLITE_FALLBACK = True
DATABASE_URL = None
engine = None
SessionLocal = None
Base = declarative_base()

def try_create_engine():
    global DATABASE_URL, engine, SessionLocal, USING_SQLITE_FALLBACK
    if DB_USER and DB_PASS and DB_NAME:
        try:
            DATABASE_URL = f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}'
            eng = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
            conn = eng.connect(); conn.close()
            engine = eng
            USING_SQLITE_FALLBACK = False
            return
        except Exception as e:
            try:
                st.sidebar.warning(f"Could not connect to MySQL using provided creds — falling back to SQLite. Error: {e}")
            except Exception:
                pass
    DATABASE_URL = 'sqlite:///smartbizlq_local.db'
    engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
    USING_SQLITE_FALLBACK = True

try_create_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# -----------------------------
# DB Models
# -----------------------------
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), default='Viewer')
    phone = Column(String(50), nullable=True)
    score = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    schema_json = Column(Text)
    rows_count = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class ModelRun(Base):
    __tablename__ = 'model_runs'
    id = Column(Integer, primary_key=True)
    model_name = Column(String(255))
    version = Column(String(100))
    metrics_json = Column(Text)
    params_json = Column(Text)
    file_path = Column(String(512))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class OTPReset(Base):
    __tablename__ = 'otp_resets'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), index=True, nullable=False)
    token_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)

class UserBadge(Base):
    __tablename__ = 'user_badges'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    badge = Column(String(255))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Create tables
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    try:
        st.sidebar.warning(f"DB create_all failed; falling back to SQLite. Error: {e}")
    except Exception:
        pass
    DATABASE_URL = 'sqlite:///smartbizlq_local.db'
    engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

# -----------------------------
# Utilities: CSV reading & cleaning
# -----------------------------
def safe_read_csv(path_or_buffer):
    try:
        if isinstance(path_or_buffer, str):
            with open(path_or_buffer, 'rb') as f:
                raw_bytes = f.read()
            raw = raw_bytes.decode('utf-8', errors='replace')
            sample = raw[:10000]
            sep = None
            if sample.count('\t') > sample.count(','):
                sep = '\t'
            elif sample.count(';') > sample.count(','):
                sep = ';'
            else:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';','|'])
                    sep = dialect.delimiter
                except Exception:
                    sep = ','
            df = pd.read_csv(io.StringIO(raw), sep=sep, engine='python')
        else:
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
            raw_bytes = path_or_buffer.read()
            raw = raw_bytes.decode('utf-8', errors='replace') if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
            sample = raw[:10000]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';','|'])
                sep = dialect.delimiter
            except Exception:
                if '\t' in sample and sample.count('\t') > sample.count(','):
                    sep = '\t'
                elif ';' in sample and sample.count(';') > sample.count(','):
                    sep = ';'
                else:
                    sep = ','
            df = pd.read_csv(io.StringIO(raw), sep=sep, engine='python')
        df.columns = [str(c).strip() for c in df.columns]
        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        for c in obj_cols:
            df[c] = df[c].astype(str).replace('nan', np.nan)
            df[c] = df[c].str.replace(r'[\r\n]+', ' ', regex=True)
            df[c] = df[c].str.replace('\t', ' ', regex=False)
            df[c] = df[c].str.strip()
        for c in df.columns:
            if df[c].dtype == object:
                sample_vals = df[c].dropna().astype(str).head(200)
                if len(sample_vals) == 0:
                    continue
                sample_clean = sample_vals.str.replace(',', '', regex=False)
                is_num = sample_clean.str.match(r'^-?\d+(\.\d+)?$')
                if is_num.sum() >= max(1, int(0.6 * len(sample_clean))):
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d\.\-]', '', regex=True).replace('', np.nan), errors='coerce')
        return df
    except Exception as e:
        try:
            st.error(f"Failed to read/clean CSV: {e}")
        except Exception:
            pass
        return pd.DataFrame()

def clean_dataframe(df: pd.DataFrame,
                    drop_empty_rows: bool = True,
                    drop_empty_cols: bool = True,
                    row_thresh: float = 0.0,
                    drop_duplicates_flag: bool = True,
                    dup_subset = None,
                    dup_keep: str = 'first'):
    if df is None:
        return pd.DataFrame()
    df2 = df.copy()
    if drop_empty_cols:
        non_empty_cols = [c for c in df2.columns if not df2[c].isnull().all() and not (df2[c].astype(str).str.strip().eq('').all())]
        df2 = df2[non_empty_cols]
    df2 = df2.replace(r'^\s*$', np.nan, regex=True)
    if drop_empty_rows:
        if row_thresh <= 0:
            df2 = df2.dropna(how='all')
        else:
            thresh = int(math.ceil((1 - row_thresh) * df2.shape[1]))
            if thresh < 1:
                thresh = 1
            df2 = df2.dropna(thresh=thresh)
    if drop_duplicates_flag:
        try:
            if dup_subset:
                df2 = df2.drop_duplicates(subset=dup_subset, keep=dup_keep)
            else:
                df2 = df2.drop_duplicates(keep=dup_keep)
        except Exception:
            df2 = df2.drop_duplicates(keep=dup_keep)
    return df2.reset_index(drop=True)

# -----------------------------
# Helpers: auth, OTP, model save
# -----------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False

def save_model(obj, name='model.joblib'):
    os.makedirs('models', exist_ok=True)
    path = os.path.join('models', f"{uuid.uuid4().hex[:8]}_{name}")
    joblib.dump(obj, path)
    return path

def create_and_store_otp(email: str, db_session, expiry_minutes: int = 15):
    token = str(secrets.randbelow(10**6)).zfill(6)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)
    rec = OTPReset(email=email, token_hash=token_hash, expires_at=expires_at, used=False)
    db_session.add(rec); db_session.commit()
    return token

def validate_otp(email: str, token: str, db_session):
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    rec = (db_session.query(OTPReset)
           .filter(OTPReset.email == email,
                   OTPReset.token_hash == token_hash,
                   OTPReset.used == False)
           .order_by(OTPReset.expires_at.desc())
           .first())
    if not rec:
        return False, 'Invalid token'
    if rec.expires_at < datetime.now(timezone.utc):
        return False, 'Token expired'
    rec.used = True
    db_session.commit()
    return True, 'OK'

def send_otp_email(recipient: str, token: str):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        raise RuntimeError('SMTP not configured in .env')
    msg = EmailMessage()
    msg['Subject'] = 'SmartBizLQ password reset code'
    msg['From'] = SMTP_USER
    msg['To'] = recipient
    msg.set_content(f'Your SmartBizLQ password reset code is: {token}\nIt expires in 15 minutes.')
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

# -----------------------------
# Session helpers: store df and detected cols
# -----------------------------
def store_df_in_session(df: pd.DataFrame, name: str = 'active_df'):
    if df is None or df.empty:
        st.session_state[name] = pd.DataFrame()
        st.session_state[f'{name}_numeric'] = []
        st.session_state[f'{name}_cat'] = []
        st.session_state[f'{name}_date'] = []
        st.session_state[f'{name}_all'] = []
        return
    st.session_state[name] = df.copy()
    all_cols = list(df.columns)
    st.session_state[f'{name}_all'] = all_cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_candidates = []
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notnull().sum() > 0.3 * len(df):
                date_candidates.append(c)
        except Exception:
            pass
    cat_cols = [c for c in all_cols if c not in numeric_cols and c not in date_candidates]
    st.session_state[f'{name}_numeric'] = numeric_cols
    st.session_state[f'{name}_cat'] = cat_cols
    st.session_state[f'{name}_date'] = date_candidates

# -----------------------------
# Visualization helpers
# -----------------------------
def safe_plotly(fig):
    try:
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.warning(f"Unable to render plot: {e}")

def compute_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    avg = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}

# -----------------------------
# Memory-safe training function (improved)
# -----------------------------
def train_churn_model(df, churn_col, feature_cols, debug=False):
    if churn_col not in df.columns:
        raise ValueError("Churn column not found in dataset")
    df_local = df[[churn_col] + [c for c in feature_cols if c in df.columns]].copy()
    y = pd.to_numeric(df_local[churn_col], errors='coerce')
    mask_valid = y.notnull()
    df_local = df_local.loc[mask_valid]
    y = y.loc[mask_valid].astype(int)
    if y.nunique() <= 1:
        raise ValueError("Target has <=1 classes after cleaning")
    X_parts = []
    for c in feature_cols:
        if c not in df_local.columns:
            continue
        if df_local[c].dtype == object or df_local[c].dtype.name == 'category':
            nunq = df_local[c].nunique(dropna=True)
            if nunq <= 20:
                X_parts.append(pd.get_dummies(df_local[c], prefix=c, dummy_na=True))
            else:
                X_parts.append(df_local[c].astype('category').cat.codes.replace(-1, np.nan).to_frame(c))
        else:
            X_parts.append(pd.DataFrame({c: pd.to_numeric(df_local[c], errors='coerce')}))
    if len(X_parts) == 0:
        raise ValueError("No features available after processing")
    X = pd.concat(X_parts, axis=1).fillna(0)
    for col in X.select_dtypes(include=[np.number]).columns:
        try:
            X[col] = X[col].astype(np.float32)
        except Exception:
            pass
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_params = dict(n_estimators=10, max_depth=8, n_jobs=1, max_features='sqrt', random_state=42, bootstrap=True)
    try:
        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred)
        return clf, metrics, 'rf_small'
    except MemoryError:
        traceback.print_exc()
        st.warning("RandomForest MemoryError — falling back")
    except Exception:
        traceback.print_exc()
        st.warning("RandomForest failed — falling back")
    try:
        sample_frac = 0.2 if len(X_train) > 50000 else 0.5 if len(X_train) > 10000 else 1.0
        if sample_frac < 1.0:
            Xs = X_train.sample(frac=sample_frac, random_state=42)
            ys = y_train.loc[Xs.index]
        else:
            Xs, ys = X_train, y_train
        lr = LogisticRegression(max_iter=200, solver='saga', n_jobs=1)
        lr.fit(Xs, ys)
        ypred = lr.predict(X_test)
        metrics = compute_classification_metrics(y_test, ypred)
        return lr, metrics, 'sampled_logistic'
    except MemoryError:
        traceback.print_exc()
        st.warning("Logistic Regression MemoryError — falling back to SGD")
    except Exception:
        traceback.print_exc()
        st.warning("Logistic Regression failed — falling back to SGD")
    try:
        sgd = SGDClassifier(max_iter=1000, tol=1e-3)
        classes = np.unique(y_train)
        sgd.partial_fit(X_train, y_train, classes=classes)
        ypred = sgd.predict(X_test)
        metrics = compute_classification_metrics(y_test, ypred)
        return sgd, metrics, 'sgd_partial'
    except Exception as ex:
        traceback.print_exc()
        raise RuntimeError(f"All training strategies failed: {ex}")

# -----------------------------
# Visuals hub (subset)
# -----------------------------
def render_visuals_hub(df):
    st.header("Visuals Hub — Many PowerBI-like visuals")
    if df is None or df.empty:
        st.warning("Upload or select a dataset first.")
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    date_cols = [c for c in df.columns if 'date' in c.lower() or np.issubdtype(df[c].dtype, np.datetime64)]
    if not date_cols:
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notnull().sum() > 0.3 * len(df):
                    date_cols.append(c)
            except Exception:
                pass
    st.sidebar.subheader("Visual controls")
    visual = st.sidebar.selectbox("Visual", [
        "Stacked Bar Chart","Clustered Bar Chart","100% Stacked Bar Chart",
        "Line Chart","Area Chart","Stacked Area Chart","Combo Chart",
        "Pie Chart","Donut Chart","Scatter Chart","Treemap",
        "Table","Matrix","Card","Multi-row Card","Gauge","KPI",
        "Waterfall Chart","Funnel Chart","Map","Choropleth","Heatmap","Histogram"
    ], index=0, key='vh_visual')
    x = st.sidebar.selectbox("X", options=[None] + cat_cols + date_cols + numeric_cols, key='vh_x')
    y = st.sidebar.selectbox("Y (numeric)", options=[None] + numeric_cols, key='vh_y')
    group = st.sidebar.selectbox("Group / color", options=[None] + cat_cols, key='vh_group')
    size = st.sidebar.selectbox("Size", options=[None] + numeric_cols, key='vh_size')
    color = st.sidebar.selectbox("Color", options=[None] + cat_cols + numeric_cols, key='vh_color')
    agg = st.sidebar.selectbox("Aggregation", options=['sum','mean','count','median'], index=0, key='vh_agg')
    top_n = st.sidebar.number_input("Top N", min_value=1, max_value=200, value=10, key='vh_topn')
    if visual in ["Stacked Bar Chart","Clustered Bar Chart","100% Stacked Bar Chart"]:
        st.subheader(visual)
        if not x or not y:
            st.info("Choose X (category) and Y (measure) in sidebar.")
        else:
            try:
                if group:
                    if "100%" in visual:
                        df_agg = df.groupby([x, group])[y].sum().reset_index()
                        df_sum = df_agg.groupby(x)[y].transform('sum')
                        df_agg['pct'] = df_agg[y] / df_sum
                        fig = px.bar(df_agg, x=x, y='pct', color=group)
                    else:
                        barmode = 'stack' if 'Stacked' in visual else 'group'
                        fig = px.bar(df, x=x, y=y, color=group, barmode=barmode)
                else:
                    fig = px.bar(df, x=x, y=y)
                safe_plotly(fig)
            except Exception as e:
                st.warning(f"Failed to render bar: {e}")
    elif visual in ["Line Chart","Area Chart","Stacked Area Chart","Combo Chart"]:
        st.subheader(visual)
        if not x or not y:
            st.info("Select X and Y")
        else:
            try:
                if visual == "Line Chart":
                    fig = px.line(df, x=x, y=y, color=group, markers=True)
                elif visual == "Area Chart":
                    fig = px.area(df, x=x, y=y, color=group)
                elif visual == "Stacked Area Chart":
                    fig = px.area(df, x=x, y=y, color=group, line_group=group)
                else:
                    agg_ts = df.groupby(x)[y].sum().reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=agg_ts[x], y=agg_ts[y], name='Bar'))
                    fig.add_trace(go.Scatter(x=agg_ts[x], y=agg_ts[y], name='Line', yaxis='y2'))
                    fig.update_layout(yaxis2=dict(overlaying='y', side='right'))
                safe_plotly(fig)
            except Exception as e:
                st.warning(f"Failed to render line/area/combo: {e}")
    elif visual in ["Pie Chart","Donut Chart"]:
        st.subheader(visual)
        if not x or not y:
            st.info("Select category and measure")
        else:
            fig = px.pie(df, names=x, values=y, hole=0.4 if visual == "Donut Chart" else 0.0)
            safe_plotly(fig)
    elif visual == "Scatter Chart":
        st.subheader("Scatter Chart")
        if not x or not y:
            st.info("Select numeric X & Y")
        else:
            fig = px.scatter(df, x=x, y=y, color=color, size=size, hover_data=df.columns)
            safe_plotly(fig)
    elif visual == "Treemap":
        st.subheader("Treemap")
        if not x:
            st.info("Select a categorical X")
        else:
            fig = px.treemap(df, path=[x] + ([group] if group else []), values=y)
            safe_plotly(fig)
    elif visual == "Table":
        st.subheader("Table")
        st.dataframe(df, width='stretch')
    elif visual == "Matrix":
        st.subheader("Matrix (pivot)")
        rows = st.sidebar.selectbox("Rows", options=[None] + cat_cols, key='vh_matrix_rows')
        cols = st.sidebar.selectbox("Cols", options=[None] + cat_cols, key='vh_matrix_cols')
        vals = st.sidebar.selectbox("Values", options=[None] + numeric_cols, key='vh_matrix_vals')
        if rows and cols and vals:
            pivot = pd.pivot_table(df, values=vals, index=rows, columns=cols, aggfunc='sum', fill_value=0)
            st.dataframe(pivot, width='stretch')
        else:
            st.info("Pick rows, columns and values for matrix.")
    elif visual in ["Card","Multi-row Card","Gauge","KPI"]:
        st.subheader(visual)
        if not y:
            st.info("Choose measure Y")
        else:
            total = float(df[y].sum(skipna=True))
            if visual == "Card":
                st.metric(f"Total {y}", f"{total:,.0f}")
            elif visual == "Multi-row Card":
                c1, c2, c3 = st.columns(3)
                c1.metric("Sum", f"{total:,.0f}")
                c2.metric("Mean", f"{df[y].mean():.2f}")
                c3.metric("Rows", len(df))
            elif visual == "Gauge":
                target = st.sidebar.number_input("Target", value=total*1.1 if total>0 else 1000.0, key='vh_gauge_target')
                fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=total, delta={'reference': float(target)}, gauge={'axis':{'range':[None, float(max(target,total,1))]}}))
                safe_plotly(fig)
            else:
                st.metric(f"KPI {y}", f"{total:,.0f}")
    elif visual in ["Waterfall Chart","Funnel Chart","Ribbon Chart"]:
        st.subheader(visual)
        if not x or not y:
            st.info("Choose X & Y")
        else:
            agg = df.groupby(x)[y].sum().reset_index().sort_values(y, ascending=False).head(top_n)
            if visual == "Waterfall Chart":
                fig = go.Figure(go.Waterfall(x=agg[x].astype(str), y=agg[y], measure=['relative']*len(agg)))
                safe_plotly(fig)
            elif visual == "Funnel Chart":
                fig = go.Figure(go.Funnel(y=agg[x].astype(str), x=agg[y]))
                safe_plotly(fig)
            else:
                if date_cols and y:
                    dc = date_cols[0]
                    df[dc] = pd.to_datetime(df[dc], errors='coerce')
                    ts = df.groupby(pd.Grouper(key=dc, freq='M'))[y].sum().reset_index()
                    fig = px.area(ts, x=dc, y=y)
                    safe_plotly(fig)
                else:
                    st.info("Ribbon approximates over time; provide a date column.")
    elif visual in ["Map","Choropleth"]:
        st.subheader(visual)
        geo_cols = [c for c in df.columns if any(k in c.lower() for k in ['country','region','state','location'])]
        if not geo_cols or not y:
            st.info("Need a geo-like column and measure")
        else:
            geo = geo_cols[0]
            agg_geo = df.groupby(geo)[y].sum().reset_index()
            try:
                fig = px.choropleth(agg_geo, locations=geo, locationmode='country names', color=y)
                safe_plotly(fig)
            except Exception:
                st.warning("Choropleth failed, fallback to bar")
                safe_plotly(px.bar(agg_geo.sort_values(y, ascending=False).head(top_n), x=geo, y=y))
    elif visual == "Heatmap":
        st.subheader("Heatmap (correlation)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True)
            safe_plotly(fig)
        else:
            st.info("Need 2+ numeric columns")
    elif visual == "Histogram":
        st.subheader("Histogram")
        if not y:
            st.info("Choose a numeric Y")
        else:
            bins = st.sidebar.slider("Bins", 5, 200, 30, key='vh_hist_bins')
            fig = px.histogram(df, x=y, nbins=bins)
            safe_plotly(fig)
    else:
        st.info("This visual will be implemented — pick another for now.")
    st.markdown("---")
    st.write("Controls for the visual are on the left (X/Y/Group/Size/Color/Agg/TopN). Use the Date column (if available) to pre-filter data.")

# decomposition tree, key influencers, smart narrative (from earlier)
def decomposition_tree(df, target_col, cat_cols, depth=3, top_n=5):
    if not cat_cols or depth == 0 or df.empty:
        return {}
    best_col = None
    best_score = -np.inf
    for c in cat_cols:
        try:
            agg = df.groupby(c)[target_col].sum()
            score = agg.var() if not np.isnan(agg.var()) else 0.0
            if score > best_score:
                best_score = score; best_col = c
        except Exception:
            continue
    if best_col is None:
        return {}
    node = {'column': best_col, 'children': []}
    groups = df.groupby(best_col)[target_col].sum().reset_index().sort_values(target_col, ascending=False).head(top_n)
    for _, r in groups.iterrows():
        val = r[best_col]
        sub = df[df[best_col] == val]
        node['children'].append({'value': val, 'sum': float(r[target_col]), 'subtree': decomposition_tree(sub, target_col, [c for c in cat_cols if c != best_col], depth-1, top_n)})
    return node

def key_influencers(df, target, features, problem_type='classification'):
    res = []
    try:
        X = pd.get_dummies(df[features]).fillna(0)
        y = df[target]
        if problem_type == 'classification':
            if y.nunique() <= 1:
                return []
            model = RandomForestClassifier(n_estimators=20, n_jobs=1, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=20, n_jobs=1, random_state=42)
        mask = y.notnull()
        if mask.sum() < 10:
            return []
        model.fit(X.loc[mask], y.loc[mask])
        try:
            pi = permutation_importance(model, X.loc[mask].sample(min(500, mask.sum()), random_state=1), y.loc[mask].sample(min(500, mask.sum()), random_state=1), n_repeats=8, random_state=1)
            pi_scores = sorted(zip(X.columns, pi.importances_mean), key=lambda x:-x[1])[:20]
            res = [{'feature': f, 'importance': float(s)} for f, s in pi_scores]
        except Exception:
            res = [{'feature': f, 'importance': float(s)} for f, s in sorted(zip(X.columns, model.feature_importances_), key=lambda x:-x[1])[:20]]
    except Exception:
        traceback.print_exc()
    return res

def smart_narrative(df, focus_cols=None, question=None):
    """Generate a plain-English data summary without external API calls."""
    rows = len(df); cols = len(df.columns)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=['object','category']).columns.tolist()
    insights = [f"Dataset has {rows:,} rows and {cols} columns."]
    if numeric:
        insights.append(f"Numeric columns: {', '.join(numeric[:6])}.")
    if cat:
        insights.append(f"Categorical columns: {', '.join(cat[:6])}.")
    for c in numeric[:3]:
        s = df[c].dropna()
        if not s.empty:
            insights.append(f"{c}: mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}")
    return "\n".join(insights)

# -----------------------------
# PPTX export helper
# -----------------------------
def export_dashboard_to_pptx(kpis: dict, charts: list, sample_df: pd.DataFrame, out_path: str):
    """
    kpis: dict of KPI title -> value
    charts: list of (title, png-bytes)
    sample_df: sample pandas DataFrame to include
    out_path: path to write pptx
    """
    if Presentation is None:
        raise RuntimeError("python-pptx not installed. pip install python-pptx")
    prs = Presentation()
    # Title slide
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "SmartBizLQ — Analytics Report"
    subtitle.text = f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S (UTC)')}"
    # KPI slide
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    left = Inches(0.5)
    top = Inches(0.4)
    tx = slide.shapes.add_textbox(left, top, Inches(9), Inches(1.5)).text_frame
    tx.text = "KPIs"
    for k, v in kpis.items():
        p = tx.add_paragraph()
        p.text = f"{k}: {v}"
    # Charts slides (each chart own slide)
    os.makedirs('temp_report_assets', exist_ok=True)
    for title_text, png_bytes in charts:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        try:
            slide.shapes.title.text = title_text
        except Exception:
            pass
        img_path = os.path.join('temp_report_assets', f"{uuid.uuid4().hex[:8]}.png")
        with open(img_path, 'wb') as f:
            f.write(png_bytes)
        try:
            slide.shapes.add_picture(img_path, Inches(0.5), Inches(1.0), width=Inches(9.0))
        except Exception:
            pass
    # Sample data slide (first 10 rows)
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    try:
        slide.shapes.title.text = "Sample data (first 10 rows)"
    except Exception:
        pass
    tbl_df = sample_df.head(10)
    tf = slide.shapes.add_textbox(Inches(0.3), Inches(1.0), Inches(9.0), Inches(4.5)).text_frame
    tf.text = tbl_df.to_string(index=False)
    prs.save(out_path)
    # cleanup temp assets
    try:
        import shutil
        shutil.rmtree('temp_report_assets')
    except Exception:
        pass
    return out_path

# -----------------------------
# Streamlit UI: main
# -----------------------------
st.set_page_config(page_title='SmartBizLQ', layout='wide')
with st.sidebar:
    st.title("SmartBizLQ")
    st.markdown("DB: " + ("SQLite (fallback)" if USING_SQLITE_FALLBACK else "MySQL (connected)"))
    auth_choice = st.selectbox("Choose", ["Login", "Sign up", "Forgot password"], key='sidebar_auth_choice')

if 'user' not in st.session_state:
    st.session_state.user = None

# Helper DB functions
def create_user(email, pw, name, role='Viewer', phone=None):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first():
            return False, 'Email exists'
        u = User(email=email, password_hash=hash_password(pw), full_name=name, role=role, phone=phone)
        db.add(u); db.commit()
        return True, 'User created'
    except Exception as e:
        return False, f"DB error: {e}"
    finally:
        db.close()

def authenticate_user(email, pw):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        if not u: return False, 'User not found'
        if verify_password(pw, u.password_hash):
            return True, u
        return False, 'Invalid password'
    finally:
        db.close()

# Auth flows
if auth_choice == 'Sign up':
    st.header("Sign up")
    su_email = st.text_input("Email", key='su_email')
    su_name = st.text_input("Full name", key='su_name')
    su_phone = st.text_input("Phone (optional)", key='su_phone')
    su_pw = st.text_input("Password", type='password', key='su_pw')
    su_role = st.selectbox("Role", ['Viewer', 'Analyst'], key='su_role')
    if st.button("Create account", key='su_create'):
        ok, msg = create_user(su_email, su_pw, su_name, su_role, su_phone)
        if ok: st.success(msg)
        else: st.error(msg)

elif auth_choice == 'Login':
    st.header("Login")
    li_email = st.text_input("Email", key='li_email')
    li_pw = st.text_input("Password", type='password', key='li_pw')
    if st.button("Login", key='li_login'):
        ok, res = authenticate_user(li_email, li_pw)
        if ok:
            u = res
            st.session_state.user = {'id': u.id, 'email': u.email, 'role': u.role, 'name': u.full_name}
            st.success("Logged in")
        else:
            st.error(res)

else:
    st.header("Reset password (OTP)")
    rp_email = st.text_input("Registered email", key='rp_email')
    if st.button("Send OTP", key='rp_send'):
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == rp_email).first()
            if not user:
                st.error("Email not found")
            else:
                token = create_and_store_otp(rp_email, db)
                try:
                    send_otp_email(rp_email, token)
                    st.success("OTP sent to email")
                except Exception as ex:
                    st.warning(f"SMTP failed: {ex}")
                    if os.getenv('DEV_MODE', 'false').lower() == 'true':
                        st.info(f"Development OTP (local use): {token}")
                    else:
                        st.info("OTP could not be sent. Check SMTP settings in .env.")
        finally:
            db.close()
    rp_token = st.text_input("OTP", key='rp_token')
    rp_new = st.text_input("New password", type='password', key='rp_new')
    if st.button("Reset password", key='rp_reset'):
        if not rp_email or not rp_token or not rp_new:
            st.error("Provide email, OTP and new password")
        else:
            db = SessionLocal()
            try:
                ok, msg = validate_otp(rp_email, rp_token, db)
                if not ok:
                    st.error(msg)
                else:
                    u = db.query(User).filter(User.email == rp_email).first()
                    if not u:
                        st.error("User not found (unexpected)")
                    else:
                        u.password_hash = hash_password(rp_new); db.commit(); st.success("Password reset successful")
            finally:
                db.close()

# Main app after login
if st.session_state.user:
    st.sidebar.markdown(f"**Signed in as**: {st.session_state.user['name']} ({st.session_state.user['role']})")
    if st.sidebar.button("Logout", key='sidebar_logout'):
        st.session_state.user = None

    page = st.selectbox("App page", ["Home", "Upload Data", "Dashboard", "Visuals Hub", "ML Studio", "Admin"], key='main_page')

    if page == "Home":
        st.header("SmartBizLQ — Home")
        st.write("Upload datasets, explore Visuals Hub, train models, and export reports.")
        with SessionLocal() as db:
            try:
                user = db.query(User).get(st.session_state.user['id'])
                st.metric('Score', user.score if user else 0)
                badges = db.query(UserBadge).filter(UserBadge.user_id == st.session_state.user['id']).all()
                if badges:
                    st.write('Badges:')
                    for b in badges:
                        st.write(f'- {b.badge} ({b.created_at.astimezone(timezone.utc).date()})')
            except Exception:
                pass

    elif page == "Upload Data":
        st.header("Upload CSV dataset")
        uploaded = st.file_uploader("Choose CSV file", type=['csv'], key='uploader')
        if uploaded:
            uploaded.seek(0)
            df = safe_read_csv(uploaded)
            if df is None or df.empty:
                st.error("File could not be parsed or is empty")
            else:
                # store raw uploaded in session (not saved to disk yet)
                st.session_state['uploaded_df'] = df.copy()
                st.session_state['uploaded_filename'] = uploaded.name

                st.subheader(f"Preview (first 20 rows) — {uploaded.name}")
                st.dataframe(df.head(20), width='stretch')
                st.write("Missing values per column:")
                st.write(df.isnull().sum()[df.isnull().sum() > 0])
                st.write("Duplicate rows:", df.duplicated().sum())

                # cleaning options
                st.markdown("### Cleaning options (preview)")
                drop_empty_rows = st.checkbox("Drop rows that are all/mostly empty", value=True, key='clean_drop_rows')
                drop_empty_cols = st.checkbox("Drop columns that are entirely empty", value=True, key='clean_drop_cols')
                drop_dups = st.checkbox("Drop duplicate rows", value=True, key='clean_drop_dups')
                row_thresh = st.slider("Row emptiness threshold (fraction empty to drop)", 0.0, 0.9, 0.0, 0.05, key='clean_row_thresh')

                cleaned = clean_dataframe(df,
                                          drop_empty_rows=drop_empty_rows,
                                          drop_empty_cols=drop_empty_cols,
                                          row_thresh=row_thresh,
                                          drop_duplicates_flag=drop_dups,
                                          dup_subset=None,
                                          dup_keep='first')
                st.subheader("Preview after cleaning")
                st.dataframe(cleaned.head(20), width='stretch')
                st.write("Original rows:", len(df), "→ Cleaned rows:", len(cleaned), "Columns:", len(cleaned.columns))

                # store in session so Visuals/ Dashboard can use it without saving
                if st.button("Use this cleaned dataset for this session", key='use_cleaned_session'):
                    store_df_in_session(cleaned, name='active_df')
                    st.session_state['uploaded_df'] = cleaned.copy()
                    st.success("Cleaned dataset stored in session as the current upload (unsaved). You can now select 'Session: Current upload (unsaved)' in Dashboard/Visuals/ML Studio.")

                if st.button("Save cleaned dataset to datasets/ folder", key='save_cleaned'):
                    os.makedirs('datasets', exist_ok=True)
                    path = os.path.join('datasets', uploaded.name)
                    uploaded.seek(0)
                    cleaned.to_csv(path, index=False)
                    with SessionLocal() as db:
                        d = Dataset(filename=uploaded.name, schema_json=cleaned.dtypes.astype(str).to_json(), rows_count=len(cleaned))
                        db.add(d); db.commit()
                    st.success("Saved cleaned dataset")

        else:
            st.info("Upload a CSV file here. After uploading you can preview, clean, and either use it for this session or save it to the datasets folder.")

    elif page == "Dashboard":
        st.header("Dashboard")
        ds_files = []
        if 'uploaded_filename' in st.session_state and 'uploaded_df' in st.session_state:
            ds_files.append("Session: Current upload (unsaved)")
        ds_files += [f for f in os.listdir('datasets')] if os.path.exists('datasets') else []

        if not ds_files:
            st.info("No datasets found. Upload CSV in Upload Data first.")
        else:
            sel = st.selectbox("Dataset (select)", ds_files, key='dashboard_dataset_select')
            if sel:
                if sel == "Session: Current upload (unsaved)":
                    df = st.session_state.get('uploaded_df', pd.DataFrame()).copy()
                else:
                    path = os.path.join('datasets', sel)
                    df = safe_read_csv(path)
                if df is None or df.empty:
                    st.error("Failed to load dataset or dataset is empty.")
                else:
                    cleaned = clean_dataframe(df, drop_empty_rows=True, drop_empty_cols=True, row_thresh=0.0, drop_duplicates_flag=False)
                    store_df_in_session(cleaned, name='active_df')

                    st.markdown("**Dataset sample (cleaned)**")
                    st.dataframe(cleaned.head(10), width='stretch')

                    cols = cleaned.columns.tolist()
                    def find(cols, patterns):
                        for p in patterns:
                            for c in cols:
                                try:
                                    if p.lower() in c.lower(): return c
                                except Exception:
                                    continue
                        return None

                    date_col = find(cols, ['date','day','ds','timestamp','time'])
                    revenue_col = find(cols, ['revenue','sales','amount','total','net'])
                    qty_col = find(cols, ['quantity','qty','units','volume'])
                    category_col = find(cols, ['category','product','item','sku'])
                    region_col = find(cols, ['country','region','state','location'])
                    id_col = find(cols, ['id','order','invoice'])

                    work = cleaned.copy()
                    if date_col:
                        work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
                    if revenue_col:
                        work[revenue_col] = pd.to_numeric(work[revenue_col], errors='coerce')
                    if qty_col:
                        work[qty_col] = pd.to_numeric(work[qty_col], errors='coerce')

                    st.sidebar.subheader('Dashboard Filters (page)')
                    if date_col:
                        try:
                            min_d = work[date_col].min().date() if work[date_col].notnull().any() else datetime.now(timezone.utc).date()
                            max_d = work[date_col].max().date() if work[date_col].notnull().any() else datetime.now(timezone.utc).date()
                        except Exception:
                            min_d = datetime.now(timezone.utc).date(); max_d = datetime.now(timezone.utc).date()
                        dr = st.sidebar.date_input('Date range', [min_d, max_d], key='dash_date_range')
                    else:
                        dr = None

                    if region_col:
                        regions = sorted(work[region_col].dropna().astype(str).unique().tolist())
                        region_sel = st.sidebar.multiselect('Region / Country', ['All'] + regions, default=['All'], key='dash_region_sel')
                    else:
                        region_sel = ['All']

                    if category_col:
                        cats = sorted(work[category_col].dropna().astype(str).unique().tolist())
                        cat_sel = st.sidebar.multiselect('Category', ['All'] + cats, default=['All'], key='dash_cat_sel')
                    else:
                        cat_sel = ['All']

                    target_rev = st.sidebar.number_input('Revenue target (gauge)', min_value=0.0, value=float(work[revenue_col].sum() if (revenue_col in work.columns and work[revenue_col].notnull().any()) else 0.0), key='dash_target_rev')
                    st.sidebar.markdown('---')

                    filtered = work.copy()
                    if date_col and dr:
                        try:
                            start = pd.to_datetime(dr[0])
                            end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                            filtered[date_col] = pd.to_datetime(filtered[date_col], errors='coerce')
                            mask = (filtered[date_col] >= start) & (filtered[date_col] <= end)
                            mask = mask.fillna(False)
                            filtered = filtered.loc[mask]
                        except Exception:
                            pass
                    if region_col and 'All' not in region_sel:
                        filtered = filtered[filtered[region_col].astype(str).isin([r for r in region_sel if r != 'All'])]
                    if category_col and 'All' not in cat_sel:
                        filtered = filtered[filtered[category_col].astype(str).isin([c for c in cat_sel if c != 'All'])]

                    # KPI strip
                    kpi_cols = st.columns(4)
                    total_rev = 0.0
                    if revenue_col in filtered.columns:
                        filtered[revenue_col] = pd.to_numeric(filtered[revenue_col], errors='coerce')
                        if filtered[revenue_col].notnull().any():
                            total_rev = float(filtered[revenue_col].sum(skipna=True))
                    total_qty = 0
                    if qty_col in filtered.columns:
                        filtered[qty_col] = pd.to_numeric(filtered[qty_col], errors='coerce')
                        total_qty = int(filtered[qty_col].sum(skipna=True)) if filtered[qty_col].notnull().any() else 0
                    avg_price = (total_rev / total_qty) if (total_qty and not np.isnan(total_rev)) else np.nan
                    orders = len(filtered) if id_col is None else (filtered[id_col].nunique() if id_col in filtered.columns else len(filtered))

                    kpi_cols[0].metric("Total Revenue", f"{total_rev:,.0f}" if total_rev else "0")
                    kpi_cols[1].metric("Total Orders/Rows", f"{orders:,}")
                    kpi_cols[2].metric("Total Units", f"{total_qty:,}" if total_qty else "0")
                    kpi_cols[3].metric("Avg Price", f"{avg_price:,.2f}" if not np.isnan(avg_price) else "N/A")

                    st.markdown("---")
                    row1_col1, row1_col2, row1_col3 = st.columns([3,2,2])
                    chart_assets = []
                    with row1_col1:
                        st.subheader("Revenue Trend")
                        if date_col and revenue_col and filtered[date_col].notnull().any() and filtered[revenue_col].notnull().any():
                            freq = 'D' if filtered[date_col].nunique() > 60 else 'M'
                            ts = filtered.groupby(pd.Grouper(key=date_col, freq=freq))[revenue_col].sum().reset_index()
                            fig_ts = px.line(ts, x=date_col, y=revenue_col, title="Revenue over time", markers=True)
                            safe_plotly(fig_ts)
                            try:
                                png_bytes = fig_ts.to_image(format='png', scale=2)
                                chart_assets.append(("Revenue trend", png_bytes))
                            except Exception:
                                pass
                        else:
                            st.info("Provide valid Date + Revenue columns.")
                        st.markdown("**Recent top rows**")
                        sort_col = revenue_col if (revenue_col in filtered.columns and filtered[revenue_col].notnull().any()) else filtered.columns[0]
                        st.dataframe(filtered.sort_values(by=sort_col, ascending=False).head(8), width='stretch')

                    with row1_col2:
                        st.subheader("Top Categories")
                        if category_col and revenue_col and category_col in filtered.columns:
                            topc = filtered.groupby(category_col)[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False).head(12)
                            fig_bar = px.bar(topc, x=category_col, y=revenue_col, title="Top categories by revenue")
                            safe_plotly(fig_bar)
                            try:
                                png_bytes = fig_bar.to_image(format='png', scale=2)
                                chart_assets.append(("Top categories", png_bytes))
                            except Exception:
                                pass
                        else:
                            st.info("Category column not detected or revenue missing.")
                        if 'topc' in locals():
                            sel_cat = st.selectbox("Drill: select category", options=["(all)"] + topc[category_col].astype(str).tolist(), key='dash_sel_cat')
                            if sel_cat and sel_cat != "(all)":
                                drilled = filtered[filtered[category_col].astype(str) == sel_cat]
                                st.write(f"Drilled rows: {len(drilled)}")
                                st.dataframe(drilled.head(10), width='stretch')

                    with row1_col3:
                        st.subheader("Share & Target")
                        if revenue_col and category_col and revenue_col in filtered.columns:
                            share = filtered.groupby(category_col)[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False).head(8)
                            fig_donut = px.pie(share, names=category_col, values=revenue_col, hole=0.45, title="Revenue share (top categories)")
                            safe_plotly(fig_donut)
                            try:
                                png_bytes = fig_donut.to_image(format='png', scale=2)
                                chart_assets.append(("Revenue share (top cats)", png_bytes))
                            except Exception:
                                pass
                        else:
                            st.info("Category+Revenue required.")
                        st.markdown("**Revenue vs Target**")
                        fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=total_rev, delta={'reference': float(target_rev)}, gauge={'axis':{'range':[None, float(max(target_rev, total_rev, 1))]}}))
                        safe_plotly(fig)
                        try:
                            png_bytes = fig.to_image(format='png', scale=2)
                            chart_assets.append(("Revenue vs Target", png_bytes))
                        except Exception:
                            pass

                    btm_col1, btm_col2, btm_col3 = st.columns([3,2,2])
                    with btm_col1:
                        st.subheader("Geo / Region Map")
                        if region_col and revenue_col and region_col in filtered.columns:
                            agg_geo = filtered.groupby(region_col)[revenue_col].sum().reset_index()
                            try:
                                fig_map = px.choropleth(agg_geo, locations=region_col, locationmode='country names', color=revenue_col, title="Revenue by Country")
                                safe_plotly(fig_map)
                                try:
                                    png_bytes = fig_map.to_image(format='png', scale=2)
                                    chart_assets.append(("Revenue by geo", png_bytes))
                                except Exception:
                                    pass
                            except Exception:
                                st.warning("Map failed; showing top regions instead.")
                                bar_fig = px.bar(agg_geo.sort_values(revenue_col, ascending=False).head(10), x=region_col, y=revenue_col)
                                safe_plotly(bar_fig)
                                try:
                                    png_bytes = bar_fig.to_image(format='png', scale=2)
                                    chart_assets.append(("Top regions", png_bytes))
                                except Exception:
                                    pass
                        else:
                            st.info("Provide region & revenue columns for map.")

                    with btm_col2:
                        st.subheader("Mini KPIs")
                        if 'ts' in locals() and not ts.empty:
                            recent = ts.tail(12)
                            safe_plotly(px.area(recent, x=date_col, y=revenue_col))
                            try:
                                png_bytes = px.area(recent, x=date_col, y=revenue_col).to_image(format='png', scale=2)
                                chart_assets.append(("Mini trend", png_bytes))
                            except Exception:
                                pass
                        else:
                            st.info("Date+Revenue needed for mini KPIs.")
                        if revenue_col in filtered.columns:
                            hist_fig = px.histogram(filtered, x=revenue_col, nbins=30)
                            safe_plotly(hist_fig)
                            try:
                                png_bytes = hist_fig.to_image(format='png', scale=2)
                                chart_assets.append(("Revenue distribution", png_bytes))
                            except Exception:
                                pass

                    with btm_col3:
                        st.subheader("Details / Drill-through")
                        search_query = st.text_input("Search (text in any column)", key='dash_search')
                        if search_query:
                            mask = pd.Series(False, index=filtered.index)
                            for c in filtered.columns:
                                try:
                                    mask = mask | filtered[c].astype(str).str.contains(search_query, case=False, na=False)
                                except Exception:
                                    mask = mask | filtered[c].astype(str).str.contains(search_query, case=False, na=False)
                            table_view = filtered[mask].copy()
                        else:
                            table_view = filtered.copy()
                        st.markdown(f"**Rows available after filters:** {len(table_view)}")
                        show_all = st.checkbox("Show all rows (disable pagination)", value=False, key='dash_show_all')
                        default_page_size = 50
                        page_size = st.number_input("Rows per page", min_value=10, max_value=2000, value=default_page_size, step=10, key='dash_page_size')
                        total_rows = len(table_view)
                        if total_rows == 0:
                            st.info("No rows to display.")
                        else:
                            if show_all:
                                start_idx = 0
                                end_idx = total_rows
                                current_page = 1
                                total_pages = 1
                            else:
                                total_pages = max(1, math.ceil(total_rows / page_size))
                                current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key='dash_page_num')
                                start_idx = (current_page - 1) * page_size
                                end_idx = min(start_idx + page_size, total_rows)
                            view = table_view.iloc[start_idx:end_idx]
                            st.markdown(f"Showing rows **{start_idx}** — **{end_idx-1}** (page {current_page} of {total_pages})")
                            st.dataframe(view, width='stretch')
                            st.markdown("**Select rows**")
                            sel_mode = st.radio("Selection mode", ["None", "Select by index (from current view)", "Select by column value (from current view)"], index=0, key='dash_sel_mode')
                            selected_df = pd.DataFrame()
                            if sel_mode == "Select by index (from current view)":
                                index_options = [str(i) for i in view.index.tolist()]
                                preview_cols = view.columns.tolist()[:2]
                                if preview_cols:
                                    labeled_options = [f"{i} | " + " | ".join([str(view.loc[int(i), c])[:40] for c in preview_cols]) for i in index_options]
                                    pick = st.multiselect("Pick row indices (multi-select)", labeled_options, key='dash_pick_indices')
                                    picked_indices = []
                                    for p in pick:
                                        try:
                                            idx = int(p.split('|')[0].strip())
                                            picked_indices.append(idx)
                                        except Exception:
                                            pass
                                else:
                                    picked = st.multiselect("Pick row indices (multi-select)", index_options, key='dash_pick_indices_simple')
                                    picked_indices = [int(x) for x in picked]
                                if st.button("Select all rows in current view", key='dash_select_all_view'):
                                    picked_indices = [int(x) for x in index_options]
                                if picked_indices:
                                    selected_df = table_view.loc[picked_indices]
                            elif sel_mode == "Select by column value (from current view)":
                                col_choice = st.selectbox("Column to select by", options=view.columns.tolist(), key='dash_sel_col')
                                if col_choice:
                                    unique_vals = view[col_choice].dropna().astype(str).unique().tolist()
                                    selected_vals = st.multiselect(f"Choose {col_choice} values (from current page)", options=unique_vals, key='dash_sel_vals')
                                    if st.button("Select all rows with chosen values (current page)", key='dash_select_vals_all'):
                                        selected_vals = unique_vals
                                    if selected_vals:
                                        selected_df = table_view[table_view[col_choice].astype(str).isin(selected_vals)].copy()
                            if not selected_df.empty:
                                st.markdown(f"**Selected rows:** {len(selected_df)}")
                                st.dataframe(selected_df, width='stretch')
                                csv_bytes = selected_df.to_csv(index=False).encode('utf-8')
                                st.download_button("Download selected rows (CSV)", csv_bytes, file_name=f"selected_rows_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", key='dash_download_selected')
                                idx_list = selected_df.index.tolist()
                                st.text_area("Selected row indices (copy/paste)", value=", ".join(map(str, idx_list)), height=80, key='dash_selected_indices_text')

                    st.markdown("---")
                    st.write("Tip: Use the **sidebar boxes** (Dataset, Date range, Region, Category, Revenue target) to control the dashboard filters and cross-filter all charts.")

                    # EXPORT: Build PPTX report from collected chart assets + KPIs + sample table
                    if st.button("Export report (PPTX)", key='export_pptx'):
                        try:
                            kpis = {
                                "Total Revenue": f"{total_rev:,.2f}",
                                "Total Orders/Rows": f"{orders:,}",
                                "Total Units": f"{total_qty:,}",
                                "Avg Price": f"{avg_price:.2f}" if not np.isnan(avg_price) else "N/A"
                            }
                            sample_df = filtered.head(100)
                            out_dir = 'reports'
                            os.makedirs(out_dir, exist_ok=True)
                            fname = f"SmartBizLQ_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pptx"
                            out_path = os.path.join(out_dir, fname)
                            # If no chart assets collected, attempt to render a few defaults and add them
                            if not chart_assets:
                                try:
                                    # attempt to generate a few charts to include
                                    if date_col and revenue_col and filtered[date_col].notnull().any() and filtered[revenue_col].notnull().any():
                                        ts = filtered.groupby(pd.Grouper(key=date_col, freq='M'))[revenue_col].sum().reset_index()
                                        fig_ts = px.line(ts, x=date_col, y=revenue_col, title="Revenue over time", markers=True)
                                        chart_assets.append(("Revenue trend", fig_ts.to_image(format='png', scale=2)))
                                    if category_col and revenue_col and category_col in filtered.columns:
                                        topc = filtered.groupby(category_col)[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False).head(10)
                                        fig_bar = px.bar(topc, x=category_col, y=revenue_col, title="Top categories by revenue")
                                        chart_assets.append(("Top categories", fig_bar.to_image(format='png', scale=2)))
                                    if revenue_col in filtered.columns:
                                        hist_fig = px.histogram(filtered, x=revenue_col, nbins=30)
                                        chart_assets.append(("Revenue distribution", hist_fig.to_image(format='png', scale=2)))
                                except Exception:
                                    pass
                            export_dashboard_to_pptx(kpis, chart_assets, sample_df, out_path)
                            with open(out_path, 'rb') as f:
                                data = f.read()
                            st.success(f"Report exported: {out_path}")
                            st.download_button("Download PPTX", data, file_name=fname, mime='application/vnd.openxmlformats-officedocument.presentationml.presentation')
                        except Exception as e:
                            st.error(f"Export failed: {e}")
                            st.exception(e)

    elif page == "Visuals Hub":
        st.header("Visuals Hub")
        ds_files = []
        if 'uploaded_filename' in st.session_state and 'uploaded_df' in st.session_state:
            ds_files.append("Session: Current upload (unsaved)")
        ds_files += [f for f in os.listdir('datasets')] if os.path.exists('datasets') else []
        if not ds_files:
            st.info('Upload a CSV dataset first.')
        else:
            sel = st.selectbox("Choose dataset", ds_files, key='vh_dataset_select')
            if sel:
                if sel == "Session: Current upload (unsaved)":
                    df = st.session_state.get('uploaded_df', pd.DataFrame()).copy()
                else:
                    df = safe_read_csv(os.path.join('datasets', sel))
                render_visuals_hub(df)

    elif page == "ML Studio":
        st.header("ML Studio")
        ds_files = []
        if 'uploaded_filename' in st.session_state and 'uploaded_df' in st.session_state:
            ds_files.append("Session: Current upload (unsaved)")
        ds_files += [f for f in os.listdir('datasets')] if os.path.exists('datasets') else []
        if not ds_files:
            st.info("Upload a dataset first.")
        else:
            sel = st.selectbox("Choose dataset", ds_files, key='ml_dataset_select')
            if sel:
                if sel == "Session: Current upload (unsaved)":
                    df = st.session_state.get('uploaded_df', pd.DataFrame()).copy()
                else:
                    df = safe_read_csv(os.path.join('datasets', sel))
                if df is None or df.empty:
                    st.error("Dataset empty or unreadable.")
                else:
                    st.dataframe(df.head(6))
                    mode = st.selectbox("Mode", ["Forecasting", "Churn Prediction", "Clustering"], key='ml_mode')
                    if mode == "Forecasting":
                        st.write("Prophet forecasting (demo)")
                        if Prophet is None:
                            st.info("Prophet not installed. pip install prophet")
                        else:
                            date_col = st.selectbox("Date column", df.columns, key='ml_forecast_date')
                            numeric_choices = df.select_dtypes(include=[np.number]).columns.tolist()
                            target_col = st.selectbox("Target column", numeric_choices, key='ml_forecast_target')
                            periods = st.number_input("Forecast periods", min_value=1, max_value=365, value=30, key='ml_forecast_periods')
                            if st.button("Train forecast", key='ml_forecast_train'):
                                ts = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
                                ts['ds'] = pd.to_datetime(ts['ds'], errors='coerce')
                                ts['y'] = pd.to_numeric(ts['y'], errors='coerce')
                                ts = ts.dropna(subset=['ds','y'])
                                if ts.empty:
                                    st.error('No valid time series rows after cleaning.')
                                else:
                                    m = Prophet()
                                    m.fit(ts)
                                    future = m.make_future_dataframe(periods=periods)
                                    forecast = m.predict(future)
                                    safe_plotly(px.line(forecast, x='ds', y='yhat', title='Forecast'))
                                    path = save_model(m, name='prophet.joblib')
                                    with SessionLocal() as db:
                                        mr = ModelRun(model_name='prophet', version='v1', metrics_json=json.dumps({'info':'trained'}), params_json=json.dumps({'periods': periods}), file_path=path)
                                        db.add(mr); db.commit()
                                    st.success('Forecast trained & saved')
                    elif mode == "Churn Prediction":
                        st.write("Churn classifier (demo)")
                        churn_col = st.selectbox('Choose churn column (binary 0/1)', df.columns, key='ml_churn_col')
                        feature_cols = st.multiselect('Select features', [c for c in df.columns if c != churn_col], key='ml_churn_feats')
                        if st.button('Train classifier', key='ml_churn_train'):
                            if not feature_cols:
                                st.error('Select at least one feature.')
                            else:
                                try:
                                    model, metrics, strategy = train_churn_model(df, churn_col, feature_cols, debug=True)
                                    st.success(f'Trained using strategy: {strategy}')
                                    st.json(metrics)
                                    path = save_model(model, name=f'churn_{strategy}.joblib')
                                    with SessionLocal() as db:
                                        mr = ModelRun(model_name='churn', version=strategy, metrics_json=json.dumps(metrics), params_json=json.dumps({'features': feature_cols}), file_path=path)
                                        db.add(mr); db.commit()
                                        ub = UserBadge(user_id=st.session_state.user['id'], badge='Trained churn model')
                                        db.add(ub)
                                        user = db.query(User).get(st.session_state.user['id'])
                                        if user:
                                            user.score += 50
                                        db.commit()
                                    st.success('Classifier trained & saved')
                                except Exception as e:
                                    st.error(f"Training failed: {e}")
                                    st.exception(e)
                    elif mode == "Clustering":
                        st.write("Clustering (KMeans)")
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if not numeric_cols:
                            st.warning('No numeric columns available for clustering')
                        else:
                            n_clusters = st.slider('Clusters', 2, 10, 3, key='ml_kmeans_clusters')
                            if st.button('Run Clustering', key='ml_kmeans_run'):
                                from sklearn.cluster import KMeans
                                X = df[numeric_cols].fillna(0)
                                k = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
                                df['cluster'] = k.labels_
                                st.dataframe(df['cluster'].value_counts())
                                path = save_model(k, name='kmeans.joblib')
                                with SessionLocal() as db:
                                    mr = ModelRun(model_name='kmeans', version='v1', metrics_json=json.dumps({'inertia': float(k.inertia_)}), params_json=json.dumps({'n_clusters': n_clusters}), file_path=path)
                                    db.add(mr); db.commit()
                                st.success('Clustering run saved')

    elif page == "Admin":
        if st.session_state.user['role'] != 'Admin':
            st.error("Admin only")
        else:
            st.header("Admin Console")
            with SessionLocal() as db:
                users = db.query(User).all()
                st.write("Users:")
                st.table([[u.id, u.email, u.full_name, u.role] for u in users])
                # Model run history
                runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).all()
                if runs:
                    st.write("Recent model runs:")
                    st.table([[r.id, r.model_name, r.version, r.created_at] for r in runs[:20]])

else:
    st.info("Please login or sign up to continue")

# Footer help
st.sidebar.markdown("---")
st.sidebar.markdown("SmartBizLQ — Template. Fill .env for DB/SMTP/OpenAI. Remove dev OTP exposure before production.")
