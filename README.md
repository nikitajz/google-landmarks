# Google Landmark Recognition 2020
https://www.kaggle.com/c/landmark-recognition-2020/

Install required packages:
  
    conda create --name landmarks --file requirements.txt python=3.8 --channel conda-forge
    conda activate landmarks
    conda install -y albumentations efficient-pytorch joblib kaggle matplot
lib numpy pandas pytorch-lightning seaborn pytorch torchvision wandb
    conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
    conda install -y faiss-gpu cudatoolkit=10.2 -c pytorch # For CUDA10
