import requests
import json
import time
import subprocess
import os
import sys

def test_api():
    print("Starting Flask backend test...")
    
    # Start backend in a separate process
    backend_proc = subprocess.Popen([sys.executable, 'app.py'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
    time.sleep(5)  # Give it time to start
    
    test_reviews = [
        "The food was amazing and delivery was super fast! Loved it.",
        "The delivery was late and the food was cold. Very disappointed.",
        "It was okay, nothing special but the delivery was on time."
    ]
    
    url = 'http://localhost:5000/predict'
    
    for review in test_reviews:
        try:
            print(f"\nTesting review: {review}")
            response = requests.post(url, json={'review': review})
            if response.status_code == 200:
                print(f"Result: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")
            
    # Terminate backend
    backend_proc.terminate()
    print("\nBackend test complete.")

if __name__ == "__main__":
    test_api()
