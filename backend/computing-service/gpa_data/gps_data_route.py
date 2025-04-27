from typing import List

from fastapi import APIRouter, HTTPException

from gpa_data.dto.gps_data_creator import GpsDataCreator
from gpa_data.gps_data_model import GpsData
from gpa_data.gps_data_service import GpsDataServiceDependency

gps_router = APIRouter()

@gps_router.get("/gps")
def read_all(service: GpsDataServiceDependency) -> List[GpsData]:
    gps_data_list: List[GpsData] = service.get_all()
    return gps_data_list

@gps_router.get("/gps/{gps_id}", response_model=GpsData)
def read_gps(gps_id: str, service: GpsDataServiceDependency):
    return service.get_one(gps_id)

@gps_router.post("/gps")
def create_gps(gps: GpsDataCreator, service: GpsDataServiceDependency) -> None:
    service.create_gps_data(gps)