from typing import Annotated, List

from fastapi import Depends

from configs.database import SessionDependency
from gpa_data.gps_data_model import GpsData


class GpsDataRepository:
    def __init__(self, session: SessionDependency) -> None:
        self.session = session

    def find_all(self) -> List[GpsData]:
        return self.session.query(GpsData).all()

    def find_one(self, gps_id: str) -> GpsData | None:
        return self.session.get(GpsData, gps_id)

    def create_one(self, gps_data: GpsData) -> None:
        self.session.add(gps_data)
        self.session.commit()
        self.session.refresh(gps_data)


GpsDataRepositoryDependency = Annotated[GpsDataRepository, Depends(GpsDataRepository)]