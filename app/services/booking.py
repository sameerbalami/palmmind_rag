from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.booking import Booking
from app.schemas.booking import BookingCreate


class BookingConflictError(ValueError):
    pass


class BookingService:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, data: BookingCreate) -> Booking:
        existing = self._db.scalar(
            select(Booking).where(Booking.email == data.email, Booking.date == data.date, Booking.time == data.time)
        )
        if existing is not None:
            raise BookingConflictError("An interview is already booked for this email at that date and time")

        booking = Booking(**data.model_dump())
        self._db.add(booking)
        self._db.commit()
        return booking

    def find_upcoming(self, session_id: str, email: str | None = None) -> list[Booking]:
        """Bookings made in this chat session, or under `email`, that have not happened yet."""
        owner = Booking.session_id == session_id
        if email:
            owner = or_(owner, func.lower(Booking.email) == email.lower())
        now = datetime.now()
        bookings = self._db.scalars(
            select(Booking).where(owner, Booking.date >= now.date()).order_by(Booking.date, Booking.time)
        )
        return [b for b in bookings if datetime.combine(b.date, b.time) > now]

    def list_all(self) -> list[Booking]:
        return list(self._db.scalars(select(Booking).order_by(Booking.date, Booking.time)))
