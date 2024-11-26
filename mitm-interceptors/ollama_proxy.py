import json
from mitmproxy import http


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and update the options in the request payload.
    """
    try:
        # Decode the content from bytes to a JSON object
        content = json.loads(flow.request.data.content.decode('utf-8'))

        # Modify the options in the JSON object
        content['options'] = {
            "seed": -1,
            "stream": True,
            "num_batch": 512,
            "num_thread": 4,
            "temperature": 0.7,
            "num_ctx": 4096,
            "num_predict": -1,
            "use_mmap": True,
            "use_mlock": False,
            "num_gpu": 0,
            "num_keep": 0,
        }

        # Convert the JSON object back to bytes
        flow.request.data.content = json.dumps(content).encode('utf-8')

    except json.JSONDecodeError:
        # Handle cases where the request body is not JSON
        print("Request content is not JSON-decodable")


# Commands
# mitmdump -s mitm-interceptors/ollama_proxy.py --mode reverse:http://jetairm1:11434 -p 11434
