from starlette.requests import Request
from starlette.responses import JSONResponse

from gpa_data.exceptions import GpsDataNotFoundException
from main import app


@app.exception_handler(GpsDataNotFoundException)
def gps_data_not_found_exception_handler(request: Request, exception: GpsDataNotFoundException):
    return JSONResponse(
        status_code=404,
        content={"message": f"Gps data {exception.gps_id} not found"}
    )