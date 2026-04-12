#!/bin/bash

# Define the base conda path and environment name
# Try to detect conda base path, fallback to common locations
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
else
    # Fallback to common locations
    if [ -d "/opt/anaconda3" ]; then
        CONDA_BASE="/opt/anaconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    else
        CONDA_BASE="/home/${USER}/anaconda3"
    fi
fi
ENV_NAME="sapiens"
PYTHON_VERSION="3.10"
# Detect OS and set PyTorch version accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - no CUDA support
    PYTORCH_VERSION=""
else
    # Linux - with CUDA
    PYTORCH_VERSION="pytorch-cuda=12.1"
fi

# Update with the path to your local conda directory
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Function to check if conda environment exists
conda_env_exists() {
  conda env list | grep -q "$1"
}

# Function to print messages in green
print_green() {
  echo -e "\033[0;32m$1\033[0m"
}

# Remove the environment if it exists
if conda_env_exists "${ENV_NAME}"; then
  print_green "Environment '${ENV_NAME}' exists. Removing..."
  conda env remove -n "${ENV_NAME}"
fi

# Create the new environment and activate it
print_green "Creating environment '${ENV_NAME}'..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"

# Ensure pip is available
print_green "Installing pip..."
conda install pip -y

# Install fish terminal
print_green "Installing fish terminal..."
conda install -c conda-forge fish -y

# Install PyTorch, torchvision, torchaudio, and specific CUDA version
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_green "Installing PyTorch, torchvision, torchaudio for macOS..."
    conda install pytorch torchvision torchaudio -c pytorch -y
else
    print_green "Installing PyTorch, torchvision, torchaudio, and CUDA (conda metapackage)..."
    conda install pytorch torchvision torchaudio "${PYTORCH_VERSION}" -c pytorch -c nvidia -y
    # Conda PyTorch on Linux often hits ImportError: undefined symbol iJIT_NotifyEvent (MKL/ITT).
    # Official cu121 wheels avoid that; use the same stack mmcv builds against.
    print_green "Replacing conda PyTorch with pip wheels (CUDA 12.1)..."
    python -m pip uninstall -y torch torchvision torchaudio torchtriton 2>/dev/null || true
    python -m pip install 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'torchaudio==2.5.1+cu121' \
        --index-url https://download.pytorch.org/whl/cu121
    # Toolchain + headers for building mmcv ops (matches PyTorch 12.1; avoids system GCC 13 + CUDA 13 nvcc issues)
    print_green "Installing CUDA 12.1 toolchain and GCC 12 for mmcv extension builds..."
    conda install -y \
        cuda-nvcc=12.1 cuda-cudart-dev=12.1 'cuda-cccl=12.1.109' 'cuda-libraries-dev=12.1' \
        'gcc_linux-64=12.*' 'gxx_linux-64=12.*' \
        -c nvidia -c conda-forge
fi

# setuptools>=82 can break legacy setup.py (missing pkg_resources during isolated builds)
print_green "Pinning setuptools for editable installs..."
python -m pip install 'setuptools>=69,<82'

# Install additional Python packages
print_green "Installing additional Python packages..."
python -m pip install scipy munkres tqdm cython fsspec yapf==0.40.1 matplotlib packaging omegaconf ipdb ftfy regex
python -m pip install chumpy --no-build-isolation
python -m pip install json_tricks terminaltables modelindex prettytable albumentations libcom
if [[ "$OSTYPE" != "darwin"* ]]; then
    # libcom may pull a newer torch; keep training stack on 2.5.1+cu121
    python -m pip install 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'torchaudio==2.5.1+cu121' \
        --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
fi

# Change directory to the root of the repository
cd "$(dirname "$0")/.."

# Function to install a package via pip with editable mode and verbose output
pip_install_editable() {
  print_green "Installing $1..."
  cd "$1" || exit
  if [[ "$OSTYPE" != "darwin"* ]]; then
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
    # mmcv compile can oversubscribe; cap parallel jobs
    export MAX_JOBS="${MAX_JOBS:-8}"
  fi
  # --no-build-isolation: avoids setuptools/pkg_resources issues in pip's build env
  python -m pip install -e . --no-build-isolation -v
  cd - || exit
  print_green "Finished installing $1."
}

# Install engine
pip_install_editable "engine"

# Install cv, handling dependencies
pip_install_editable "cv"
python -m pip install -r "cv/requirements/optional.txt"  # Install optional requirements

# Install pretrain
pip_install_editable "pretrain"

# Install pose
pip_install_editable "pose"

# Install det
pip_install_editable "det"

# Install seg
pip_install_editable "seg"

print_green "Installation done!"
