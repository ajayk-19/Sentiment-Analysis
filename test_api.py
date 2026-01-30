import requests
import json
import sys
import io

# Set encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

url = "http://127.0.0.1:5000/predict"
reviews = [
    "The food was absolutely delicious and arrived much earlier than expected! 😊👌",
    "Great experience! The packaging was perfect and no spills. ❤️😍",
    "I'm very disappointed. The order was cold and the delivery person was rude. 😒😡",
    "App kept crashing while I was trying to pay. Very frustrating. 😒",
    "Super fast delivery! 👍🙌"
]

for review in reviews:
    print(f"\nAnalyzing Review: '{review}'")
    try:
        response = requests.post(url, json={"review": review})
        if response.status_code == 200:
            result = response.json()
            print(f"Sentiment: {result['sentiment']}")
            if 'probabilities' in result:
                print(f"Probabilities: {result['probabilities']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
