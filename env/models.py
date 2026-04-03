from typing import Any

from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from openenv.core.env_server.types import State
from pydantic import Field


class EnterpriseEmailAction(OpenEnvAction):
    category: str = Field(..., description="Predicted email category")
    priority: str = Field(..., description="Predicted email priority")
    route: str = Field(..., description="Team the email should be routed to")
    response: str = Field(..., description="Agent response sent to the customer")
    extracted_entities: dict[str, str] = Field(
        default_factory=dict,
        description="Structured entities extracted from the email",
    )


class EnterpriseEmailObservation(OpenEnvObservation):
    email: str = Field(default="", description="Current customer email")
    history: list[str] = Field(
        default_factory=list,
        description="Responses generated during the current episode",
    )
    extracted_entities: dict[str, str] = Field(
        default_factory=dict,
        description="Latest extracted entities associated with the episode",
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra environment info such as score and termination reason",
    )


class EnterpriseEmailState(State):
    task_name: str = Field(default="easy", description="Active task identifier")
    terminated: bool = Field(default=False, description="Whether the episode ended")
    last_score: float = Field(default=0.0, description="Most recent grader score")
    termination_reason: str | None = Field(
        default=None,
        description="Why the current episode terminated",
    )


Action = EnterpriseEmailAction
Observation = EnterpriseEmailObservation
