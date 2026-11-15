from datetime import date, datetime, time
from typing import Self

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class BookingCreate(BaseModel):
    session_id: str
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    date: date
    time: time

    @model_validator(mode="after")
    def _must_be_in_future(self) -> Self:
        if datetime.combine(self.date, self.time) <= datetime.now():
            raise ValueError("The interview date and time must be in the future")
        return self


class BookingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: EmailStr
    date: date
    time: time
    created_at: datetime
