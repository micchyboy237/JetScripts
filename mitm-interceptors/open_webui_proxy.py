from mitmproxy import http


def request(flow: http.HTTPFlow):
    if flow.request.path.startswith("/static"):
        flow.request.port = 8081


# Commands
# mitmdump -s mitm-interceptors/open_webui_proxy.py --mode reverse:http://jetairm1:8085 -p 8080
