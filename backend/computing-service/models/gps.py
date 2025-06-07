from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field


class BaseGPSData(SQLModel):
    """Base class for GPS Data with common fields"""
    __tablename__ = "gps_data"

    accuracy: float
    latitude: float
    longitude: float
    altitude: float
    course: float
    speed: float
    fixtime: datetime

class GPSDataInput(SQLModel):
    """Input model for GPS data (from Kafka)"""
    id: int
    accuracy: float
    altitude: float
    course: float
    fixtime: float # Unix timestamp
    latitude: float
    longitude: float
    speed: float

class GPSData(SQLModel, table=True):
    """Database model for GPS data"""
    __tablename__ = "filtered_gps"

    id: Optional[int] = Field(default=None, primary_key=True)
    external_id: str = Field(index=True)

    f_latitude: Optional[float] = None
    f_longitude: Optional[float] = None
    v_latitude: Optional[float] = None
    v_longitude: Optional[float] = None

    fixtime: Optional[datetime] = None

class GPSDataResponse(BaseGPSData):
    """Response model for GPS data"""
    id: int
    external_id: str
    f_latitude: Optional[float] = None
    f_longitude: Optional[float] = None
    v_latitude: Optional[float] = None
    v_longitude: Optional[float] = None
