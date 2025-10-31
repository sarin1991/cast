apt-get -y update
apt-get -y install git curl wget screen vim
python3 -m venv train
source train/bin/activate
pip install --upgrade pip
pip install torch numpy sentencepiece protobuf wheel datasets accelerate ninja einops hf-transfer
# MAX_JOBS=16 pip install flash-attn==2.8 --no-build-isolation
git clone https://github.com/sarin1991/cast.git
git clone https://github.com/sarin1991/transformers.git
cd transformers/
git checkout cast
pip install -e .

#NSYS

# apt update
# apt install -y --no-install-recommends gnupg
# echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
# apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# apt update
# apt install -y nsight-systems-cli