from collections.abc import Iterable
import json
from mitmproxy import http
import logging
import time
from jet.transformers import make_serializable

# Dictionary to store start times for requests
start_times = {}


def interceptor_callback(data: bytes) -> bytes | Iterable[bytes]:
    """
    This function will be called for each chunk of request/response body data that arrives at the proxy,
    and once at the end of the message with an empty bytes argument (b"").
    """
    # print(f"chunk: {data.decode('utf-8')}")
    return data


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and record the start time.
    """
    logging.info(f"URL: {flow.request.host}{flow.request.path}")
    # Log the decoded data as a JSON string
    logging.info(f"REQUEST:\n{json.dumps(
        make_serializable(flow.request.data), indent=2)}")
    start_times[flow.id] = time.time()  # Store the start time for the request


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """
    end_time = time.time()  # Record the end time
    if flow.id in start_times:
        time_taken = end_time - start_times[flow.id]
        logging.info(f"Request to {flow.request.host}{
                     flow.request.path} took {time_taken:.2f} seconds.")
        del start_times[flow.id]  # Clean up to avoid memory issues
    else:
        logging.warning(f"Start time for {flow.id} not found!")


def responseheaders(flow):
    """
    Set the response interceptor callback for streaming.
    """
    flow.response.stream = interceptor_callback


# Commands
# mitmdump -s mitm-interceptors/ollama_interceptor.py --mode reverse:http://jetairm1:11434 -p 11434
