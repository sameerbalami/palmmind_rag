from datetime import date, time

import pytest

from app.models.booking import Booking
from app.services.chat import describe_booking

TODAY = date(2026, 11, 14)


def _booking(day: date) -> Booking:
    return Booking(
        id="038a0a0a-0000-0000-0000-000000000000",
        session_id="s1",
        name="Samir Balami",
        email="samir@example.com",
        date=day,
        time=time(14, 0),
    )


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 11, 14), "(today)"),
        (date(2026, 11, 15), "(tomorrow)"),
        (date(2026, 11, 20), "(in 6 days)"),
    ],
)
def test_days_remaining(day: date, expected: str) -> None:
    assert expected in describe_booking(_booking(day), TODAY)


def test_includes_weekday_time_and_reference() -> None:
    text = describe_booking(_booking(date(2026, 11, 20)), TODAY)
    assert text.startswith("Friday, 20 November 2026 at 2:00 PM")
    assert "reference: 038a0a0a" in text
