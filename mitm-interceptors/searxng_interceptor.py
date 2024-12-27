from collections.abc import Iterable
from datetime import datetime
import json
import os
from mitmproxy import http
import time
from jet.transformers import make_serializable, format_prompt_log
from jet.logger import logger

# Dictionary to store start times for requests
start_times: dict[str, float] = {}


def request(flow: http.HTTPFlow):
    """
    Handle the request, log it, and record the start time.
    """
    if any(path == flow.request.path for path in ["/search"]):
        logger.log("\n")
        url = f"{flow.request.scheme}//{flow.request.host}{flow.request.path}"
        logger.info(f"URL: {url}")
        # Log the serialized data as a JSON string
        request_dict = make_serializable(flow.request.data)
        logger.log(f"REQUEST KEYS:", list(
            request_dict.keys()), colors=["GRAY", "INFO"])
        logger.log(f"REQUEST:")
        logger.debug(json.dumps(request_dict, indent=2))
        # Store the start time for the request
        start_times[flow.id] = time.time()


def response(flow: http.HTTPFlow):
    """
    Handle the response, calculate and log the time difference.
    """
    if any(path == flow.request.path for path in ["/search"]):
        logger.log("\n")
        # Log the serialized data as a JSON string
        response_dict = make_serializable(flow.response.data)
        logger.log(f"RESPONSE KEYS:", list(
            response_dict.keys()), colors=["GRAY", "INFO"])
        logger.log(f"RESPONSE:")
        logger.debug(json.dumps(response_dict['content'], indent=2))

        end_time = time.time()  # Record the end time

        if flow.id in start_times:
            time_taken = end_time - start_times[flow.id]
            logger.log("Request total time took:", f"{time_taken:.2f} seconds", colors=[
                "LOG",
                "BRIGHT_SUCCESS",
            ])
            del start_times[flow.id]  # Clean up to avoid memory issues
        else:
            logger.warning(f"Start time for {flow.id} not found!")
