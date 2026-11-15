import uuid
from datetime import UTC, date, datetime, time

from sqlalchemy import Date, DateTime, String, Time, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Booking(Base):
    """An interview booking captured from a chat conversation."""

    __tablename__ = "bookings"
    __table_args__ = (UniqueConstraint("email", "date", "time", name="uq_booking_email_slot"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), index=True)
    date: Mapped[date] = mapped_column(Date)
    time: Mapped[time] = mapped_column(Time)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
