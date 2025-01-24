from jet.logger import logger
from jet.transformers.formatters import format_json
from mitmproxy import http


def request(flow: http.HTTPFlow):
    logger.newline()
    logger.info("REQUEST:")
    logger.debug(format_json(flow.request))


def response(flow: http.HTTPFlow):
    logger.newline()
    logger.info("RESPONSE:")
    logger.debug(format_json(flow.response))


# Commands
# mitmdump -s mitm-interceptors/open_webui_proxy.py --mode reverse:http://jetairm1:8085 -p 8080
