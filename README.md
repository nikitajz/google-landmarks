# Google Landmark Recognition 2020
https://www.kaggle.com/c/landmark-recognition-2020/

Install required packages:
  
    conda create --name landmarks python=3.8
    conda activate landmarks
    conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
    conda install -y albumentations efficientnet-pytorch joblib kaggle matplotlib numpy pandas pretrainedmodels pytorch-lightning seaborn scikit-learn wandb
    conda install -y faiss-cpu
    # for GPU version use below
    # conda install -y faiss-gpu cudatoolkit=10.2 -c pytorch # For CUDA10.2
    
Note that CUDA version (here 10.2) should correspond to your system-wide installed.
