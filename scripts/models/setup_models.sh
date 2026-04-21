#!/usr/bin/env bash
# ============================================================================
# RoboRenForce — VLA Model Setup Script
#
# Downloads and configures VLA model weights for all supported backbones.
# Run from the project root: bash scripts/models/setup_models.sh [model_name]
#
# Usage:
#   bash scripts/models/setup_models.sh           # Interactive: choose model
#   bash scripts/models/setup_models.sh qwen2vl   # Setup Qwen2-VL
#   bash scripts/models/setup_models.sh qwen3vl   # Setup Qwen3-VL
#   bash scripts/models/setup_models.sh openpi     # Setup OpenPI (pi0/pi0.5)
#   bash scripts/models/setup_models.sh gr00t      # Setup GR00T N1.7
#   bash scripts/models/setup_models.sh all        # Setup all models
# ============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
MODELS_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

activate_venv() {
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
        info "Using venv: ${VENV_DIR}"
    else
        warn "No venv found at ${VENV_DIR}, using system Python"
    fi
}

# ============================================================================
# Qwen2-VL
# ============================================================================
setup_qwen2vl() {
    info "Setting up Qwen2-VL..."
    echo ""
    echo "  Model:   Qwen/Qwen2-VL-2B-Instruct (4.1GB)"
    echo "  Also:    Qwen/Qwen2-VL-7B-Instruct (15GB)"
    echo "  Requires: transformers>=4.37, qwen-vl-utils"
    echo ""

    activate_venv

    # Install dependencies
    pip install -q "transformers>=4.37" "qwen-vl-utils" "accelerate" 2>/dev/null || true

    # Download model weights (HuggingFace auto-caches)
    python -c "
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import sys

model_name = 'Qwen/Qwen2-VL-2B-Instruct'
print(f'Downloading {model_name}...')
try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype='auto', trust_remote_code=True,
    )
    print(f'  Hidden size: {model.config.hidden_size}')
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B')
    del model, processor
    print('Download complete!')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
"
    ok "Qwen2-VL ready"
}

# ============================================================================
# Qwen3-VL
# ============================================================================
setup_qwen3vl() {
    info "Setting up Qwen3-VL..."
    echo ""
    echo "  Model:   Qwen/Qwen3-VL-2B-Instruct (4.5GB)"
    echo "  Also:    Qwen/Qwen3-VL-8B-Instruct (16GB)"
    echo "  Requires: transformers>=4.51, qwen-vl-utils"
    echo ""

    activate_venv

    pip install -q "transformers>=4.51" "qwen-vl-utils" "accelerate" 2>/dev/null || true

    python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import sys

model_name = 'Qwen/Qwen3-VL-2B-Instruct'
print(f'Downloading {model_name}...')
try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype='auto', trust_remote_code=True,
    )
    print(f'  Hidden size: {model.config.hidden_size}')
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B')
    del model, processor
    print('Download complete!')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
"
    ok "Qwen3-VL ready"
}

# ============================================================================
# OpenPI (pi0 / pi0.5)
# ============================================================================
setup_openpi() {
    info "Setting up OpenPI (pi0/pi0.5)..."
    echo ""
    echo "  Option A — Via LeRobot (recommended):"
    echo "    Model:   lerobot/pi05_base (4B params)"
    echo "    Install: pip install 'lerobot[pi]@git+https://github.com/huggingface/lerobot.git'"
    echo ""
    echo "  Option B — Native OpenPI:"
    echo "    Repo:    https://github.com/Physical-Intelligence/openpi"
    echo "    Install: git clone + uv sync"
    echo "    Checkpoints on GCS: gs://openpi-assets/checkpoints/"
    echo ""

    activate_venv

    # Try LeRobot path first
    info "Installing via LeRobot (HuggingFace path)..."
    if pip install -q "lerobot[pi]@git+https://github.com/huggingface/lerobot.git" 2>/dev/null; then
        ok "LeRobot + pi0 installed"

        # Download model weights
        python -c "
try:
    from lerobot.policies.pi05 import PI05Policy
    print('Downloading lerobot/pi05_base...')
    policy = PI05Policy.from_pretrained('lerobot/pi05_base')
    print('Download complete!')
    del policy
except ImportError:
    try:
        from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
        print('Downloading lerobot/pi05_base...')
        policy = PI05Policy.from_pretrained('lerobot/pi05_base')
        print('Download complete!')
        del policy
    except Exception as e:
        print(f'Warning: Could not download model: {e}')
        print('You can download later when first using the model.')
" 2>&1 || true
    else
        warn "LeRobot install failed. Trying native OpenPI..."

        OPENPI_DIR="${PROJECT_ROOT}/third_party/openpi"
        if [ ! -d "${OPENPI_DIR}" ]; then
            info "Cloning OpenPI..."
            git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "${OPENPI_DIR}"
        fi

        if command -v uv &>/dev/null; then
            cd "${OPENPI_DIR}" && GIT_LFS_SKIP_SMUDGE=1 uv sync
            ok "Native OpenPI installed"
        else
            warn "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
            warn "Then run: cd ${OPENPI_DIR} && uv sync"
        fi
    fi

    ok "OpenPI setup complete"
}

# ============================================================================
# GR00T N1.7
# ============================================================================
setup_gr00t() {
    info "Setting up NVIDIA GR00T N1.7..."
    echo ""
    echo "  Option A — Via HuggingFace (download weights only):"
    echo "    Model:   nvidia/GR00T-N1.7-3B (6GB)"
    echo "    Also:    nvidia/GR00T-N1.7-DROID, nvidia/GR00T-N1.7-LIBERO"
    echo "    Requires: transformers, flash-attn"
    echo ""
    echo "  Option B — Full Isaac-GR00T package:"
    echo "    Repo:    https://github.com/NVIDIA/Isaac-GR00T"
    echo "    Install: git clone + uv sync --python 3.10"
    echo "    Requires: CUDA 12.6+, flash-attn 2.7.4+"
    echo ""

    activate_venv

    # Install HuggingFace deps
    pip install -q "transformers>=4.40" "accelerate" "safetensors" 2>/dev/null || true

    # Try to install flash-attn (may fail without CUDA)
    pip install -q flash-attn 2>/dev/null || warn "flash-attn not installed (needs CUDA build)"

    # Download model weights
    python -c "
from transformers import AutoModel, AutoProcessor
import sys

model_name = 'nvidia/GR00T-N1.7-3B'
print(f'Downloading {model_name}...')
try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        print(f'  Hidden size: {model.config.hidden_size}')
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B')
    del model, processor
    print('Download complete!')
except Exception as e:
    print(f'Warning: Could not download via HF: {e}')
    print('The model may require accepting terms at: https://huggingface.co/nvidia/GR00T-N1.7-3B')
    print('Or install the full Isaac-GR00T package for native loading.')
" 2>&1 || true

    echo ""
    info "For full Isaac-GR00T package (optional):"
    echo "  git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T third_party/isaac-gr00t"
    echo "  cd third_party/isaac-gr00t && uv sync --python 3.10"

    ok "GR00T setup complete"
}

# ============================================================================
# Main
# ============================================================================
show_menu() {
    echo ""
    echo "=== RoboRenForce — VLA Model Setup ==="
    echo ""
    echo "Available models:"
    echo "  1) qwen2vl   — Qwen2-VL 2B/7B      (Alibaba, Apache 2.0)"
    echo "  2) qwen3vl   — Qwen3-VL 2B/8B      (Alibaba, Apache 2.0)"
    echo "  3) openpi    — pi0/pi0.5 4B         (Physical Intelligence, Apache 2.0)"
    echo "  4) gr00t     — GR00T N1.7 3B        (NVIDIA, Apache 2.0)"
    echo "  5) all       — Setup all models"
    echo ""
    echo "Usage: bash scripts/models/setup_models.sh [model_name]"
    echo ""
}

MODEL="${1:-}"

if [ -z "${MODEL}" ]; then
    show_menu
    read -p "Choose model (1-5 or name): " choice
    case "${choice}" in
        1|qwen2vl)  MODEL="qwen2vl" ;;
        2|qwen3vl)  MODEL="qwen3vl" ;;
        3|openpi)   MODEL="openpi"  ;;
        4|gr00t)    MODEL="gr00t"   ;;
        5|all)      MODEL="all"     ;;
        *)          err "Unknown choice: ${choice}"; exit 1 ;;
    esac
fi

case "${MODEL}" in
    qwen2vl)  setup_qwen2vl ;;
    qwen3vl)  setup_qwen3vl ;;
    openpi)   setup_openpi  ;;
    gr00t)    setup_gr00t   ;;
    all)
        setup_qwen2vl
        setup_qwen3vl
        setup_openpi
        setup_gr00t
        ;;
    *)
        err "Unknown model: ${MODEL}"
        show_menu
        exit 1
        ;;
esac

echo ""
ok "Done! Verify with: python -c \"from RRF_models.registry import list_models; print(list_models())\""
