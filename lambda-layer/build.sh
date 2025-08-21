#!/bin/bash

set -e  # Exit on any error

# Build Lambda layer using Docker
echo "🐳 Building Lambda layer with Docker..."

# Clean up any previous builds
rm -f layer.zip
docker rmi chatbot-layer 2>/dev/null || true

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t chatbot-layer .

# Run container and extract the layer.zip
echo "📦 Extracting layer.zip..."
docker run --rm --entrypoint="" -v $(pwd):/output chatbot-layer sh -c "cp /lambda/layer.zip /output/"

# Verify the layer was created
if [ -f "layer.zip" ]; then
    echo "✅ Lambda layer built successfully!"
    echo "📁 Layer file: $(pwd)/layer.zip"
    echo "📏 Size: $(ls -lh layer.zip | awk '{print $5}')"
    
    # Show contents summary
    echo "📋 Layer contents:"
    unzip -l layer.zip | head -20
else
    echo "❌ Failed to create layer.zip"
    exit 1
fi
