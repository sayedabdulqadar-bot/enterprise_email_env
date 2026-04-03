import os

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from env.models import (
    EnterpriseEmailAction,
    EnterpriseEmailObservation,
    EnterpriseEmailState,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def get_base_url(base_url: str | None = None) -> str:
    return (
        base_url
        or os.getenv("ENTERPRISE_EMAIL_ENV_BASE_URL")
        or os.getenv("OPENENV_BASE_URL")
        or DEFAULT_BASE_URL
    )


class EnterpriseEmailEnv(
    EnvClient[
        EnterpriseEmailAction,
        EnterpriseEmailObservation,
        EnterpriseEmailState,
    ]
):
    def __init__(self, base_url: str | None = None, **kwargs):
        super().__init__(base_url=get_base_url(base_url), **kwargs)

    def _step_payload(self, action: EnterpriseEmailAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self,
        payload: dict,
    ) -> StepResult[EnterpriseEmailObservation]:
        obs_data = payload.get("observation", {})
        observation = EnterpriseEmailObservation(
            email=obs_data.get("email", ""),
            history=obs_data.get("history", []),
            extracted_entities=obs_data.get("extracted_entities", {}),
            info=obs_data.get("info", {}),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict) -> EnterpriseEmailState:
        return EnterpriseEmailState(**payload)


def create_sync_client(base_url: str | None = None):
    return EnterpriseEmailEnv(base_url=base_url).sync()


if __name__ == "__main__":
    with create_sync_client() as env:
        result = env.reset(task_name="easy")
        print("Reset:", result.observation.model_dump())

        action = EnterpriseEmailAction(
            category="billing",
            priority="medium",
            route="billing_team",
            response="We are sorry for the inconvenience and will help resolve this issue.",
            extracted_entities={"order_id": "1234"},
        )
        result = env.step(action)
        print("Step:", result.observation.model_dump())
