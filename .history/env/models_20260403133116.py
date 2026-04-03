from pydantic import BaseModel
from typing import Optional, Dict


class Observation(BaseModel):
    email: str
    history: list[str]
    extracted_entities: Dict[str, str] = {}


class Action(BaseModel):
    category: str
    priority: str
    route: str
    response: str
    extracted_entities: Dict[str, str]


class Reward(BaseModel):
    value: float