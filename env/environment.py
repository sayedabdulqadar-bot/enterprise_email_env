import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from env.graders import grade
from env.models import Action, EnterpriseEmailState, Observation
from env.reward import compute_step_reward


class EmailOpsEnv(Environment[Action, Observation, EnterpriseEmailState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS = 5
    TASK_INDEX = {
        "easy": 0,
        "medium": 1,
        "hard": 2,
    }

    def __init__(self, task_name: str = "easy"):
        super().__init__()
        self.tasks = self._load_tasks()
        self._default_task_name = task_name
        self._task_name = task_name
        self._task = self._select_task(task_name)
        self._history: list[str] = []
        self._last_entities: dict[str, str] = {}
        self._done = False
        self._prev_score = 0.0
        self._termination_reason: str | None = None
        self._state = EnterpriseEmailState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=self._task_name,
            terminated=False,
            last_score=0.0,
            termination_reason=None,
        )

    def _load_tasks(self) -> list[dict[str, Any]]:
        data_path = Path(__file__).resolve().parent.parent / "data" / "emails.json"
        with data_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _select_task(self, task_name: str) -> dict[str, Any]:
        if task_name in self.TASK_INDEX:
            return self.tasks[self.TASK_INDEX[task_name]]

        if task_name.isdigit():
            index = int(task_name)
            if 0 <= index < len(self.tasks):
                return self.tasks[index]

        valid_names = ", ".join(sorted(self.TASK_INDEX))
        raise ValueError(f"Unknown task_name '{task_name}'. Expected one of: {valid_names}")

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> Observation:
        observation_info = {
            "score": self._state.last_score,
            "task_name": self._task_name,
            "step_count": self._state.step_count,
        }
        if self._termination_reason is not None:
            observation_info["termination_reason"] = self._termination_reason
            observation_info["success"] = self._termination_reason == "success"
        if info:
            observation_info.update(info)

        return Observation(
            email=self._task["email"],
            history=list(self._history),
            extracted_entities=dict(self._last_entities),
            done=done,
            reward=reward,
            info=observation_info,
            metadata={
                "episode_id": self._state.episode_id,
                "task_name": self._task_name,
                "step_count": self._state.step_count,
            },
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: str | None = None,
        **_: Any,
    ) -> Observation:
        self._reset_rubric()
        self._task_name = task_name or self._task_name or self._default_task_name
        self._task = self._select_task(self._task_name)
        self._history = []
        self._last_entities = {}
        self._done = False
        self._prev_score = 0.0
        self._termination_reason = None
        self._state = EnterpriseEmailState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._task_name,
            terminated=False,
            last_score=0.0,
            termination_reason=None,
        )

        return self._build_observation(
            reward=0.0,
            done=False,
            info={"seed": seed} if seed is not None else None,
        )

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **_: Any,
    ) -> Observation:
        if self._done:
            return self._build_observation(
                reward=0.0,
                done=True,
                info={
                    "error": "Episode already terminated. Call reset() to start a new episode.",
                },
            )

        self._state.step_count += 1

        score = grade(action, self._task)
        reward = compute_step_reward(
            self._prev_score,
            score,
            self._state.step_count,
        )

        self._prev_score = score
        self._state.last_score = score
        self._history.append(action.response)
        self._last_entities = dict(action.extracted_entities)

        if score >= 0.9:
            self._done = True
            self._termination_reason = "success"
        elif self._state.step_count >= self.MAX_STEPS:
            self._done = True
            self._termination_reason = "max_steps"

        self._state.terminated = self._done
        self._state.termination_reason = self._termination_reason

        return self._build_observation(
            reward=reward,
            done=self._done,
            info={"timeout_s": timeout_s} if timeout_s is not None else None,
        )

    @property
    def state(self) -> EnterpriseEmailState:
        return self._state
