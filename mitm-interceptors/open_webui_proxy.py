from mitmproxy import http


def request(flow: http.HTTPFlow):
    if flow.request.path.startswith("/static"):
        flow.request.port = 8081


def response(flow: http.HTTPFlow) -> None:
    # Add CORS headers to allow all origins
    if flow.response:
        flow.response.headers["Access-Control-Allow-Origin"] = "*"
        flow.response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        flow.response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"


# Commands
# mitmdump -s mitm-interceptors/open_webui_proxy.py --mode reverse:http://localhost:8080 -p 8080
