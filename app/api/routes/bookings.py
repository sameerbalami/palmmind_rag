from fastapi import APIRouter

from app.api.deps import BookingServiceDep
from app.models.booking import Booking
from app.schemas.booking import BookingOut

router = APIRouter(prefix="/bookings", tags=["Bookings"])


@router.get("", response_model=list[BookingOut])
def list_bookings(service: BookingServiceDep) -> list[Booking]:
    return service.list_all()
