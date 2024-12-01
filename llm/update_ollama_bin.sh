#!/bin/bash

# Define GitHub API URL for Ollama releases
GITHUB_API_URL="https://api.github.com/repos/ollama/ollama/releases/latest"

# Use curl and jq to parse the JSON returned by GitHub API to find the download URL for Ollama-darwin.zip and ollama-darwin
OLLAMA_DARWIN_ZIP_URL=$(curl -s $GITHUB_API_URL | jq -r '.assets[] | select(.name | test("Ollama-darwin\\.zip$")) | .browser_download_url')
OLLAMA_DARWIN_BIN_URL=$(curl -s $GITHUB_API_URL | jq -r '.assets[] | select(.name | test("ollama-darwin$")) | .browser_download_url')

# Ensure the URLs were found
if [[ -z "$OLLAMA_DARWIN_ZIP_URL" || -z "$OLLAMA_DARWIN_BIN_URL" ]]; then
    echo "Failed to find the download URLs. Exiting..."
    exit 1
fi

# Download Ollama-darwin.zip
echo "Downloading Ollama-darwin.zip..."
curl -L $OLLAMA_DARWIN_ZIP_URL -o Ollama-darwin.zip

# Unzip Ollama-darwin.zip
echo "Unpacking Ollama-darwin.zip..."
unzip Ollama-darwin.zip

# Move the Ollama application to the user's Applications directory
echo "Moving Ollama application to the Applications directory..."
mv Ollama.app /Applications/

# Clean up the zip file
rm Ollama-darwin.zip

# Download ollama-darwin binary
echo "Downloading ollama-darwin binary..."
curl -L $OLLAMA_DARWIN_BIN_URL -o ollama-darwin

# Make the binary executable
chmod +x ollama-darwin

# Move the binary to /usr/local/bin/ollama
echo "Moving ollama-darwin to /usr/local/bin/ollama..."
sudo mv ollama-darwin /usr/local/bin/ollama

echo "Ollama installation complete."