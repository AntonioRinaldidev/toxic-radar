import requests

response = requests.post("http://localhost:8001/classify", json={
    "text": "You're the worst person I've ever met!"
})

print(response.json())
