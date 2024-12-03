#!/bin/bash

PORT="8080"
TARGET_URL="http://searxng.local:8081"
SCRIPT="searxng_interceptor.py"

# Run mitmdump in client mode
echo "Running client proxy..."
mitmdump -s $SCRIPT --mode reverse:$TARGET_URL -p $PORT
