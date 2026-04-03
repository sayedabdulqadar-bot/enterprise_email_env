import random

TEMPLATES = [
    "I need help with order #{id}",
    "Why was I charged twice for #{id}?",
    "Account hacked! urgent help",
    "Refund request for order #{id}"
]

def generate_email():
    template = random.choice(TEMPLATES)
    return template.replace("{id}", str(random.randint(1000, 9999)))