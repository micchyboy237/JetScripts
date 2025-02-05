#!/bin/bash

# Default configuration
LISTEN_PORT="8080"
TARGET_URL="http://jetairm1:8080"
PROXY_SCRIPT="open_webui_proxy.py"


echo "Running reverse proxy with CORS enabled..."

# Run mitmdump with the CORS injection script
mitmdump --mode reverse:$TARGET_URL -p $LISTEN_PORT -s $PROXY_SCRIPT

# Sample Command
# ./run_proxy_sample.sh
