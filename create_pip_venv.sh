#!/usr/bin/env bash
set -e

rm -rf ml-env
python3 -m venv ml-env
source ml-env/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA 12.6 wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Create requirements.txt on the fly
cat << EOF > requirements.txt
pandas
pyarrow
matplotlib
scikit-learn
tensorboard
EOF

# Install other packages
pip install -r requirements.txt

# Optional: remove the temporary requirements.txt file
rm requirements.txt

# Test CUDA availability
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
"

# make it executable chmod +x create_pip_venv.sh
# run ./create_pip_venv.sh