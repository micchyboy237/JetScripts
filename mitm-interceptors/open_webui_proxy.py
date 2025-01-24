from jet.logger import logger
from jet.transformers.formatters import format_json
from mitmproxy import http


def request(flow: http.HTTPFlow):
    logger.newline()
    logger.log("flow.request.path:", flow.request.path,
               colors=["WHITE", "INFO"])
    logger.log("flow.request.pretty_host:", flow.request.pretty_host,
               colors=["WHITE", "INFO"])
    logger.info("REQUEST:")
    logger.debug(format_json(flow.request))

    if flow.request.path.startswith("/static"):
        # flow.response = http.Response.make(
        #     200,  # (optional) status code
        #     b"Hello World",  # (optional) content
        #     {"Content-Type": "text/html"},  # (optional) headers
        # )
        flow.request.port = 8081
    logger.log("flow.request.port:", flow.request.port,
               colors=["WHITE", "INFO"])
    logger.log("flow.request.url:", flow.request.url,
               colors=["WHITE", "INFO"])


def response(flow: http.HTTPFlow):
    logger.newline()
    logger.info("RESPONSE:")
    logger.debug(format_json(flow.response))


# Commands
# mitmdump -s mitm-interceptors/open_webui_proxy.py --mode reverse:http://jetairm1:8085 -p 8080
