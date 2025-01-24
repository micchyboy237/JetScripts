#!/bin/bash

# Default configuration
LISTEN_PORT="8080"
TARGET_URL="http://localhost:8085"
CORS_SCRIPT="inject_cors.py"

echo "Running reverse proxy with CORS enabled..."

# Run mitmdump with the CORS injection script
mitmdump --mode reverse:$TARGET_URL -p $LISTEN_PORT -s $CORS_SCRIPT

# Sample Command
# ./run_proxy_sample.sh
