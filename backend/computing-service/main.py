from fastapi import FastAPI

from gpa_data.gps_data_route import gps_router

app = FastAPI()

app.include_router(gps_router)
