#!/bin/bash
# Setup script for RoboTwin simulator integration
#
# Usage:
#   bash source/tasks/RRF_robotwin/scripts/setup_robotwin.sh [INSTALL_DIR]
#
# This script:
#   1. Clones the RoboTwin repo (RLinf_support branch)
#   2. Installs simulator dependencies (sapien, mplib)
#   3. Downloads assets from HuggingFace (optional)
#   4. Installs the RRF_robotwin_tasks package

set -e

INSTALL_DIR="${1:-$(pwd)/third_party}"
ROBOTWIN_DIR="$INSTALL_DIR/RoboTwin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== RoboTwin Setup ==="
echo "Install dir: $INSTALL_DIR"
echo ""

# 1. Clone RoboTwin
if [ -d "$ROBOTWIN_DIR" ]; then
    echo "[1/4] RoboTwin already exists at $ROBOTWIN_DIR, pulling latest..."
    cd "$ROBOTWIN_DIR" && git pull
else
    echo "[1/4] Cloning RoboTwin (RLinf_support branch)..."
    mkdir -p "$INSTALL_DIR"
    git clone https://github.com/RoboTwin-Platform/RoboTwin -b RLinf_support "$ROBOTWIN_DIR"
fi

# 2. Install simulator dependencies
echo "[2/4] Installing simulator dependencies..."
pip install sapien==3.0.1 mplib==0.2.1 gymnasium>=0.29.1
pip install transforms3d trimesh open3d av toppra

# Patch sapien urdf_loader for UTF-8 (from RLinf install.sh)
SAPIEN_URDF=$(python -c "import sapien; import os; print(os.path.join(os.path.dirname(sapien.__file__), 'wrapper/urdf_loader.py'))" 2>/dev/null || true)
if [ -n "$SAPIEN_URDF" ] && [ -f "$SAPIEN_URDF" ]; then
    if ! grep -q "encoding='utf-8'" "$SAPIEN_URDF"; then
        echo "  Patching sapien urdf_loader.py for UTF-8..."
        sed -i "s/open(urdf_file)/open(urdf_file, encoding='utf-8')/g" "$SAPIEN_URDF"
    fi
fi

# 3. Download assets (optional, prompt user)
echo ""
echo "[3/4] RoboTwin assets (models, textures, etc.)"
echo "  Assets can be downloaded from HuggingFace: TianxingChen/RoboTwin2.0"
echo "  Set ROBOTWIN_ASSETS_PATH env var to the download location."
echo "  Skipping automatic download — run manually if needed:"
echo "    huggingface-cli download TianxingChen/RoboTwin2.0 --local-dir \$ROBOTWIN_ASSETS_PATH"
echo ""

# 4. Install task package
echo "[4/4] Installing RRF_robotwin_tasks..."
pip install -e "$SCRIPT_DIR"

# Add RoboTwin to PYTHONPATH
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Add RoboTwin to your PYTHONPATH:"
echo "  export PYTHONPATH=$ROBOTWIN_DIR:\$PYTHONPATH"
echo ""
echo "Optionally set assets path:"
echo "  export ROBOTWIN_ASSETS_PATH=/path/to/robotwin_assets"
