class GpsDataNotFoundException(Exception):
    def __init__(self, gps_id: str) -> None:
        self.gps_id = gps_id