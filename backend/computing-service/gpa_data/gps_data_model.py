from typing import Optional

from sqlmodel import SQLModel, Field
from datetime import datetime

class GpsData(SQLModel, table=True):
    __tablename__ = "gps_data"

    id: Optional[str] = Field(default=None, primary_key=True)
    accuracy: float
    altitude: float
    course: float
    fixtime: datetime
    latitude: float
    longitude: float
    speed: float
