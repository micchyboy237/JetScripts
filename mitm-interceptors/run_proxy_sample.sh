#!/bin/bash

# Default configuration
LISTEN_PORT="8080"
TARGET_URL="http://localhost:8085"

echo "Running reverse proxy..."
mitmdump --mode reverse:$TARGET_URL -p $LISTEN_PORT

# Sample Command
# ./run_proxy_sample.sh