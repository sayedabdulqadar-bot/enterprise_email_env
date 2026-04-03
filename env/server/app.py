import os

from openenv.core.env_server.http_server import create_app

from env.environment import EmailOpsEnv
from env.models import EnterpriseEmailAction, EnterpriseEmailObservation


def _max_concurrent_envs() -> int:
    raw_value = os.getenv("MAX_CONCURRENT_ENVS", "8")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 8


app = create_app(
    EmailOpsEnv,
    EnterpriseEmailAction,
    EnterpriseEmailObservation,
    env_name="enterprise_email_env",
    max_concurrent_envs=_max_concurrent_envs(),
)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    resolved_port = port or int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=resolved_port)


if __name__ == "__main__":
    main()
