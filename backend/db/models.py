"""Database models and async session management."""
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, JSON, DateTime, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from backend.config import settings


import re
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG, connect_args=_connect_args)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, index=True)
    image_name = Column(String, nullable=False)
    image_path = Column(Text, nullable=False)
    annotated_path = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    inference_time = Column(Float, nullable=False)
    total_cells = Column(Integer, default=0)
    rbc_count = Column(Integer, default=0)
    wbc_count = Column(Integer, default=0)
    platelet_count = Column(Integer, default=0)
    detections = Column(JSON, default=list)
    crop_paths = Column(JSON, default=list)
    confidence_scores = Column(JSON, default=list)
    bounding_boxes = Column(JSON, default=list)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
