import logging
from datetime import datetime

import numpy as np
from sqlmodel import Session

from configs.database import engine
from configs.kconsumer_factory import KConsumerFactory
from models import GPSDataInput, GPSData

logger = logging.getLogger(__name__)

class KalmanFilter:
    def __init__(self, jerk_var=0.3):
        """
        Constant-Acceleration Kalman Filter with reordered state:
        state x = [p_lat, v_lat, a_lat, p_lon, v_lon, a_lon]
        """
        self.jerk_var = jerk_var
        self.x = None
        self.P = None
        self.initialized = False

    def initialize(self, p_lat0, v_lat0, p_lon0, v_lon0, cov0=500):
        """Initialize the 6-element state and covariance."""
        # [position, velocity, acceleration] for lat and lon
        self.x = np.array([p_lat0, v_lat0, 0.0, p_lon0, v_lon0, 0.0], dtype=float)
        self.P = np.eye(6) * cov0
        self.initialized = True
        self.prev_v_lat = v_lat0
        self.prev_v_lon = v_lon0
        self.prev_time = None

    def predict(self, dt):
        """Predict step using constant acceleration model."""
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        # State transition matrix
        F = np.array([
            [1, dt, 0.5 * dt2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Process noise Q: jerk variance on acceleration states
        Q = np.array([
            [dt4 / 4, dt3 / 2, dt2 / 2, 0, 0, 0],
            [dt3 / 2, dt2, dt, 0, 0, 0],
            [dt2 / 2, dt, 1, 0, 0, 0],
            [0, 0, 0, dt4 / 4, dt3 / 2, dt2 / 2],
            [0, 0, 0, dt3 / 2, dt2, dt],
            [0, 0, 0, dt2 / 2, dt, 1]
        ], dtype=float) * self.jerk_var ** 2

        # predict state and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R):
        """Update step with measurement z."""
        H = np.eye(6)  # Identity matrix for full state measurement
        y = z - H @ self.x  # Measurement residual

        # Kalman Gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update equation
        self.x = self.x + K @ y

        # Covariance Update equation - use Joseph form for numerical stability
        I = np.eye(6)
        t = (I - K @ H)
        self.P = t @ self.P @ t.T + K @ R @ K.T

    def process_data_point(self, data: GPSDataInput):
        """Process a single GPS data point."""
        current_time = datetime.fromtimestamp(data.fixtime)

        # Calculate velocity components from speed and course
        rad = np.deg2rad(data.course)
        vn = data.speed * np.cos(rad)  # North component in m/s
        ve = data.speed * np.sin(rad)  # East component in m/s

        # Convert to deg/s - 111000 meters per degree of latitude
        # For longitude, need to account for latitude
        v_lat = vn / 111000.0
        v_lon = ve / (111000.0 * np.cos(np.deg2rad(data.latitude)))

        # Initialize if not already done
        if not self.initialized:
            self.initialize(data.latitude, v_lat, data.longitude, v_lon)
            self.prev_time = current_time
            f_lat = data.latitude
            f_lon = data.longitude
            return f_lat, f_lon, v_lat, v_lon

        # Calculate time step
        dt = (current_time - self.prev_time).total_seconds()
        dt = max(dt, 0.1)  # Ensure dt is at least 0.1 seconds

        # Predict
        self.predict(dt)

        # Calculate acceleration
        a_lat = (v_lat - self.prev_v_lat) / dt
        a_lon = (v_lon - self.prev_v_lon) / dt

        # Measurement noise based on GPS accuracy
        sigma_p = data.accuracy / 111000.0  # Position noise (deg)
        sigma_v = sigma_p / dt  # Velocity noise
        sigma_a = sigma_v / dt  # Acceleration noise

        # Build measurement noise covariance matrix
        R = np.diag([
            sigma_p ** 2,  # p_lat
            (sigma_v * 1.5) ** 2,  # v_lat (1.5x higher noise)
            (sigma_a * 3.0) ** 2,  # a_lat (3x higher noise)
            sigma_p ** 2,  # p_lon
            (sigma_v * 1.5) ** 2,  # v_lon
            (sigma_a * 3.0) ** 2  # a_lon
        ])

        # Measurement vector
        z = np.array([
            data.latitude,  # p_lat
            v_lat,  # v_lat
            a_lat,  # a_lat
            data.longitude,  # p_lon
            v_lon,  # v_lon
            a_lon  # a_lon
        ], dtype=float)

        # Update
        self.update(z, R)

        # Store for next iteration
        self.prev_v_lat = v_lat
        self.prev_v_lon = v_lon
        self.prev_time = current_time

        # Return filtered values
        f_lat = self.x[0]
        f_lon = self.x[3]

        print(f"v_lat = {v_lat}")
        print(f"v_lon = {v_lon}")

        return f_lat, f_lon, v_lat, v_lon

# Initialize a singleton instance of the filter
kalman_filter = KalmanFilter(jerk_var=0.01)

async def filter_gps():
    """Consume messages rom Kafka topic and process them"""
    consumer = KConsumerFactory.create_consumer("gps-data")

    await consumer.start()
    try:
        async for msg in consumer:
            print(msg.value)
            try:
                gps_data = GPSDataInput.parse_obj(msg.value)
                await process_gps_data(gps_data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    finally:
        consumer.paused()


async def process_gps_data(gps_data: GPSDataInput):
    """Process GPS data and appl Kalman filter"""

    def safe_float(value):
        return float(value) if isinstance(value, np.floating) else value

    try:
        # Apply Kalman Filter
        f_lat, f_lon, v_lat, v_lon = kalman_filter.process_data_point(gps_data)

        print(f"safe v_lat = {safe_float(v_lat)}")

        # Store in database
        with Session(engine) as session:
            try:
                # Convert Unix timestamp to datetime
                fixtime = datetime.fromtimestamp(gps_data.fixtime)

                # Create a new GPS data entry
                gps_entry = GPSData(
                    external_id=gps_data.id,
                    f_latitude=safe_float(f_lat),
                    f_longitude=safe_float(f_lon),
                    v_latitude=safe_float(v_lat),
                    v_longitude=safe_float(v_lon),
                    fixtime=fixtime
                )

                session.add(gps_entry)
                session.commit()
                logger.info(f"Processed GPS data for ID {gps_data.id}")
            except Exception as e:
                session.rollback()
                logger.error(f"Database error: {e}")

    except Exception as ex:
        logger.error(f"Error in process_gps_data: {ex}")
