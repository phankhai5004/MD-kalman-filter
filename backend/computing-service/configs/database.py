from typing import Annotated

from fastapi import Depends
from sqlmodel import create_engine, Session, SQLModel

DATABASE_URL = "postgresql://postgres:root@localhost:5432/kalman_filter"

engine = create_engine(DATABASE_URL, echo=True)

# Create tables
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# Dependency for database session
def get_db():
    with Session(engine) as session:
        yield session

SessionDependency = Annotated[Session, Depends(get_db)]