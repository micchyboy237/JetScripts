import requests
import threading
import time

stop_signal = threading.Event()


def stream_chat():
    url = "http://localhost:11434/api/chat"  # Change to actual API URL
    headers = {"Accept": "application/json"}

    body = {
        "model": "llama3.2",
        "messages": [
            {
                "role": "user",
                "content": "Why is the sky blue?",
            }
        ],
        "stream": True,
        # "keep_alive": keep_alive,
        # "template": template,
        # "raw": False,
        # "tools": tools,
        # "format": str(format) if format else None,
        "options": {
            "seed": 0,
            "temperature": 0,
            "num_keep": 0,
            "num_predict": -1,
        },
    }

    with requests.post(url, json=body, headers=headers) as response:
        for line in response.iter_lines():
            if stop_signal.is_set():
                print("Client stopping stream...")
                break  # Stop iterating, effectively stopping stream
            if line:
                print(line.decode('utf-8'))


if __name__ == "__main__":
    # Start streaming in a thread
    thread = threading.Thread(target=stream_chat)
    thread.start()

    # Simulate user stopping after 2 seconds
    time.sleep(2)
    stop_signal.set()

    # Optionally: Notify server/interceptor to stop
    # Define an endpoint for stopping
    response = requests.post("http://localhost:11434/api/chat/stop")

    thread.join()
    print("Stream fully stopped.")
    print("Response:", response)
