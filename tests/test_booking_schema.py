from datetime import date, time, timedelta

import pytest
from pydantic import ValidationError

from app.schemas.booking import BookingCreate


def _booking(**overrides: object) -> BookingCreate:
    data: dict[str, object] = {
        "session_id": "s1",
        "name": "Ram Sharma",
        "email": "ram@example.com",
        "date": date.today() + timedelta(days=1),
        "time": time(14, 30),
    }
    return BookingCreate.model_validate(data | overrides)


def test_valid_booking() -> None:
    assert _booking().email == "ram@example.com"


def test_rejects_invalid_email() -> None:
    with pytest.raises(ValidationError):
        _booking(email="not-an-email")


def test_rejects_past_date() -> None:
    with pytest.raises(ValidationError, match="future"):
        _booking(date=date.today() - timedelta(days=1))


def test_parses_llm_string_values() -> None:
    booking = _booking(date="2099-01-15", time="09:00")
    assert booking.date == date(2099, 1, 15)
    assert booking.time == time(9, 0)
