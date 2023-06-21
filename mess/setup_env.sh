# run script with
# bash mess/setup_env.sh

# Create new environment "mess"
conda create --name zegformer -y python=3.7
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zegformer

# Install ZegFormer requirements
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
pip install -r requirements.txt

# Install packages for dataset preparation
pip install gdown
pip install kaggle
pip install rasterio
pip install pandas