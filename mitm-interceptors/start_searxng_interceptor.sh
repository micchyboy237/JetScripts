#!/bin/bash

CLIENT_LISTEN_PORT="8080"
CLIENT_TARGET_URL="http://searxng.local:8082"
CLIENT_PROXY_SCRIPT="searxng_interceptor.py"

# Run mitmdump in client mode
echo "Running client proxy..."
mitmdump -s $CLIENT_PROXY_SCRIPT --mode reverse:$CLIENT_TARGET_URL -p $CLIENT_LISTEN_PORT