#!/bin/bash
# Setup script for gongjiyun.com cloud GPU instances.
# Run this after SSH into the instance.
#
# Usage:
#   bash setup_cloud.sh

set -euo pipefail

echo "=== Setting up GitHub acceleration ==="
export GHFAST="https://ghfast.top"

echo "=== Setting up HuggingFace mirror ==="
export HF_ENDPOINT="https://hf-mirror.com"
pip install -U huggingface_hub

echo "=== Cloning repo ==="
git clone ${GHFAST}/https://github.com/Jing-yilin/VARC.git
cd VARC
git checkout experiment/info-bottleneck

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Downloading ARC datasets ==="
mkdir -p raw_data
cd raw_data
git clone ${GHFAST}/https://github.com/fchollet/ARC-AGI.git
git clone ${GHFAST}/https://github.com/neoneye/arc-dataset-collection.git re_arc_raw || true
cd ..

echo "=== Downloading pretrained VARC checkpoint ==="
mkdir -p saves/offline_train_ViT
python -c "
from huggingface_hub import hf_hub_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
path = hf_hub_download(
    repo_id='lillianyhl/VARC',
    filename='checkpoint_best.pt',
    local_dir='saves/offline_train_ViT',
)
print(f'Downloaded to {path}')
"

echo "=== Setup complete ==="
echo "To train the original VARC baseline:"
echo "  bash script/offline_train_VARC_ViT.sh"
echo ""
echo "To train the info-bottleneck model:"
echo "  bash script/train_bottleneck_cloud.sh"
