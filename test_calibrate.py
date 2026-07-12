import requests
import json

url = "http://127.0.0.1:8080/v1/rag-gate/calibrate"

payload = {
    "target_coverage": 0.8,
    "samples": [
        {
            "question": "What is the capital of France?",
            "context": "Paris is the capital.",
            "correct": True,
            "logprobs": [-0.1, -0.2, -0.1]
        },
        {
            "question": "What is quantum mechanics?",
            "context": "Physics stuff.",
            "correct": False,
            "logprobs": [-1.5, -2.2, -1.8]
        },
        {
            "question": "Who wrote Hamlet?",
            "context": "Shakespeare wrote it.",
            "correct": True,
            "logprobs": [-0.3, -0.4, -0.5]
        }
    ]
}

response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(json.dumps(response.json(), indent=2))
