from mitmproxy import http


def response(flow: http.HTTPFlow) -> None:
    # Add CORS headers to allow all origins
    flow.response.headers["Access-Control-Allow-Origin"] = "*"
    flow.response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    flow.response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
