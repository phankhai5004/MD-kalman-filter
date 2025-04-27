from typing import Annotated

from fastapi import Depends
from sqlmodel import create_engine, Session

DATABASE_URL = "postgresql://postgres:root@localhost:5432/kalman_filter"

engine = create_engine(DATABASE_URL, echo=True)

def get_session():
    with Session(engine) as session:
        yield session

SessionDependency = Annotated[Session, Depends(get_session)]