import requests

URL = "https://geniussahil-traffic-ai-openenv.hf.space"

print("Checking /reset...")
r = requests.post(URL + "/reset")
print("Status:", r.status_code)
print("Response:", r.json())

print("\nChecking /step...")
r = requests.post(URL + "/step", json={"action": 1})
print("Status:", r.status_code)
print("Response:", r.json())
