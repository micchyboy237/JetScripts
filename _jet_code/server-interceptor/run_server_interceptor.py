# server/interceptor.py
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import logging
import os

# Setup log file
os.makedirs('server', exist_ok=True)
log_file = 'server/request_logs.txt'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(message)s')


class LoggingHandler(BaseHTTPRequestHandler):
    def _log_request(self):
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {self.command} {self.path} {self.request_version} from {self.client_address[0]}"
        logging.info(log_entry)

    def do_GET(self):
        self._log_request()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Logged GET request')

    def do_POST(self):
        self._log_request()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Logged POST request')

    def log_message(self, format, *args):
        # Suppress default console logging
        return


def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, LoggingHandler)
    print(f"Intercepting requests on port {port}...")
    httpd.serve_forever()


if __name__ == '__main__':
    run_server()
