import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from configs import create_db_and_tables
from services.kalman_filter import filter_gps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Kafka consumer and create database tables when the app starts."""
    # Create database and tables
    create_db_and_tables()
    logger.info("Database and tables created")

    # Start Kafka consumer
    asyncio.create_task(filter_gps())
    logger.info("Kafka consumer started")
    yield

app = FastAPI(title="GPS Tracking API with Kalman Filter", lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


