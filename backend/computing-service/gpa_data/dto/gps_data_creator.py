from datetime import datetime

from pydantic import BaseModel


class GpsDataCreator(BaseModel):
    accuracy: float
    altitude: float
    course: float
    fixtime: datetime
    latitude: float
    longitude: float
    speed: float