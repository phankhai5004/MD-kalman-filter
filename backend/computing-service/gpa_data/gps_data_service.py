from typing import Annotated, List
from uuid import uuid4

from fastapi import Depends

from gpa_data.dto.gps_data_creator import GpsDataCreator
from gpa_data.exceptions import GpsDataNotFoundException
from gpa_data.gps_data_model import GpsData
from gpa_data.gps_data_repo import GpsDataRepositoryDependency


class GpsDataService:
    def __init__(self, repository: GpsDataRepositoryDependency):
        self.repository = repository

    def get_all(self) -> List[GpsData]:
        return self.repository.find_all()

    def get_one(self, gps_id: str) -> GpsData | None:
        gps_data = self.repository.find_one(gps_id)

        if gps_data is None:
            raise GpsDataNotFoundException(gps_id)

        return gps_data

    def create_gps_data(self, gps: GpsDataCreator) -> None:
        gps_data = GpsData(**gps.model_dump())
        gps_data.id = str(uuid4())

        self.repository.create_one(gps_data)

GpsDataServiceDependency = Annotated[GpsDataService, Depends(GpsDataService)]