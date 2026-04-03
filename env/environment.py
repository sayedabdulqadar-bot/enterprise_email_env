import json
from env.models import Observation, Action
from env.graders import grade
from env.reward import compute_step_reward


class EmailOpsEnv:

    def __init__(self, task_name="easy"):
        with open("data/emails.json") as f:
            self.tasks = json.load(f)

        self.task = self.tasks[0]
        self.step_count = 0
        self.done = False
        self.prev_score = 0

    def reset(self):
        self.step_count = 0
        self.done = False
        self.prev_score = 0

        return Observation(
            email=self.task["email"],
            history=[]
        )

    def state(self):
        return {"step": self.step_count}

    def step(self, action: Action):
        self.step_count += 1

        score = grade(action, self.task)

        reward = compute_step_reward(
            self.prev_score,
            score,
            self.step_count
        )

        self.prev_score = score

        if score > 0.9 or self.step_count >= 5:
            self.done = True

        return {
            "observation": Observation(
                email=self.task["email"],
                history=[action.response]
            ),
            "reward": reward,
            "done": self.done,
            "info": {"score": score}
        }