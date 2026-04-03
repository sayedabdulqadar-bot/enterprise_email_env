import requests

BASE_URL = "http://localhost:8000"


def reset():
    response = requests.post(f"{BASE_URL}/reset", json={})
    return response.json()


def step(action):
    response = requests.post(f"{BASE_URL}/step", json=action)
    return response.json()


if __name__ == "__main__":
    print("Testing environment...")

    obs = reset()
    print("Reset:", obs)

    action = {
        "category": "billing",
        "priority": "high",
        "route": "billing_team",
        "response": "We will help resolve your issue.",
        "extracted_entities": {}
    }

    result = step(action)
    print("Step:", result)