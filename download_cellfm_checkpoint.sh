#!/bin/bash

# Download CellFM checkpoint from HuggingFace
# This script downloads the CellFM 80M pre-trained checkpoint

echo "🔽 Downloading CellFM 80M checkpoint..."

CKPT_DIR="/home/ubuntu/LLM-inference/xinze-project/cellfm"
CKPT_FILE="CellFM_80M_weight.ckpt"
CKPT_URL="https://huggingface.co/ShangguanNingyuan/CellFM/resolve/main/CellFM_80M_weight.ckpt"

mkdir -p "$CKPT_DIR"
cd "$CKPT_DIR"

if [ -f "$CKPT_FILE" ]; then
    echo "✓ Checkpoint already exists: $CKPT_FILE"
    ls -lh "$CKPT_FILE"
    exit 0
fi

echo "Downloading from: $CKPT_URL"
echo "Target: $CKPT_DIR/$CKPT_FILE"
echo ""
echo "⚠️  This is a large file (~320MB). Download may take several minutes."
echo ""

# Try wget first
if command -v wget &> /dev/null; then
    wget "$CKPT_URL" -O "$CKPT_FILE"
elif command -v curl &> /dev/null; then
    curl -L "$CKPT_URL" -o "$CKPT_FILE"
else
    echo "❌ Error: Neither wget nor curl is available"
    echo "Please install wget or curl, or manually download from:"
    echo "$CKPT_URL"
    exit 1
fi

if [ -f "$CKPT_FILE" ]; then
    echo ""
    echo "✅ Download completed!"
    ls -lh "$CKPT_FILE"
else
    echo "❌ Download failed"
    exit 1
fi




