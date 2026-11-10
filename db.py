"""Database layer (SQLAlchemy). SQLite locally, PostgreSQL on Render via DATABASE_URL."""
import json
import os
import re
import secrets
from datetime import datetime, timezone
import bcrypt
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, LargeBinary, Date, Index
from sqlalchemy.orm import declarative_base, sessionmaker

url = os.getenv("DATABASE_URL", "sqlite:///agri.db")
url = re.sub(r"^postgres(ql)?://", "postgresql+psycopg2://", url)
eng = create_engine(url, pool_pre_ping=True, pool_recycle=280)
Base = declarative_base()


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)
Sess = sessionmaker(eng, expire_on_commit=False)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(80))
    email = Column(String(120), unique=True, index=True)
    pw = Column(String(100))
    role = Column(String(12), default="farmer")
    active = Column(Boolean, default=True)
    created = Column(DateTime, default=_now)


class Farm(Base):
    """One farm profile per user: defaults used by every tool."""
    __tablename__ = "farms"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True, index=True)
    region = Column(String(30), default="east_humid")
    soil = Column(String(20), default="loamy")
    area_ha = Column(Float, default=1.0)
    crop = Column(String(30))
    sow_date = Column(Date)
    method = Column(String(12), default="flood")


class Record(Base):
    """Every analysis a user runs (soil, weather, irrigation, crop, fertilizer, scan, chat, obs)."""
    __tablename__ = "records"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    kind = Column(String(16), index=True)
    created = Column(DateTime, default=_now)
    summary = Column(String(300))
    data = Column(Text)
    blob = Column(LargeBinary)


Index("ix_records_user_kind", Record.user_id, Record.kind, Record.id)


def hp(p):
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()


def init():
    Base.metadata.create_all(eng)
    ae = os.getenv("ADMIN_EMAIL", "admin@agrixai.local").lower()
    with Sess() as s:
        if not s.query(User).filter_by(email=ae).first():
            pw = os.getenv("ADMIN_PASSWORD")
            if not pw:
                pw = secrets.token_urlsafe(12)
                print(f"[agrixai] ADMIN_PASSWORD not set. Generated one-time admin password for {ae}: {pw}", flush=True)
            s.add(User(name="Administrator", email=ae, pw=hp(pw), role="admin"))
            s.commit()


def save(user_id, kind, data, summary="", blob=None):
    with Sess() as s:
        r = Record(user_id=user_id, kind=kind, data=json.dumps(data, default=str), summary=summary[:300], blob=blob)
        s.add(r)
        s.commit()
        return r.id


def _row(r):
    return dict(id=r.id, kind=r.kind, created=r.created, summary=r.summary, data=json.loads(r.data or "{}"))


def latest(user_id, kind):
    with Sess() as s:
        r = s.query(Record).filter_by(user_id=user_id, kind=kind).order_by(Record.id.desc()).first()
        return _row(r) if r else None


def recent(user_id, kind=None, limit=50, with_blob=False):
    with Sess() as s:
        q = s.query(Record).filter_by(user_id=user_id)
        if kind:
            q = q.filter_by(kind=kind)
        out = []
        for r in q.order_by(Record.id.desc()).limit(limit).all():
            d = _row(r)
            if with_blob:
                d["blob"] = r.blob
            out.append(d)
        return out


def get_blob(user_id, rid):
    with Sess() as s:
        r = s.query(Record).filter_by(id=rid, user_id=user_id).first()
        return r.blob if r else None


def counts(user_id):
    with Sess() as s:
        rows = s.query(Record.kind).filter_by(user_id=user_id).all()
    out = {}
    for (k,) in rows:
        out[k] = out.get(k, 0) + 1
    return out


def get_farm(user_id):
    with Sess() as s:
        f = s.query(Farm).filter_by(user_id=user_id).first()
        if not f:
            f = Farm(user_id=user_id)
            s.add(f)
            s.commit()
        return f


def update_farm(user_id, **kw):
    with Sess() as s:
        f = s.query(Farm).filter_by(user_id=user_id).first() or Farm(user_id=user_id)
        for k, v in kw.items():
            setattr(f, k, v)
        s.add(f)
        s.commit()
        return f
