apt-get -y update
apt-get -y install git curl wget screen vim
python3 -m venv train
source train/bin/activate
pip install --upgrade pip
pip install torch numpy sentencepiece protobuf wheel datasets accelerate ninja einops
# MAX_JOBS=16 pip install flash-attn==2.8 --no-build-isolation
git clone https://github.com/sarin1991/cast.git
git clone https://github.com/sarin1991/transformers.git
cd transformers/
git checkout cast
pip install -e .