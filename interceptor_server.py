from fastapi import FastAPI, Request
import httpx  # Async HTTP client for making requests
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)


@app.api_route("/endpoint", methods=["GET", "POST", "PUT", "DELETE"])
async def log_and_forward(request: Request):
    # Log incoming request details
    logging.info("Intercepted Request")
    logging.info("URL: %s", str(request.url))
    logging.info("Headers: %s", dict(request.headers))
    logging.info("Params: %s", dict(request.query_params))
    logging.info("Data: %s", await request.body())

    # Prepare request data
    method = request.method
    url = "http://localhost:11434/api/generate"  # Target endpoint
    headers = dict(request.headers)
    params = dict(request.query_params)
    data = await request.body()

    # Forward the request to the actual endpoint
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

    # Return the real response to the client
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": response.text,
    }

# Don't call uvicorn.run() here. Let the command line handle this.
