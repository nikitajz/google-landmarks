My experiments based on Google Landmarks Dataset v2:  
https://github.com/cvdfoundation/google-landmark

and accompanying competition  
https://www.kaggle.com/c/landmark-recognition-2020/

Install required packages:
  
    conda create --name landmarks python=3.7
    conda activate landmarks
    pip install --use-feature=2020-resolver -r requirements.txt
    
Optionally CUDA version (here 10.2) could be installed using anaconda instead of system-wide installation:
    conda install -y cudatoolkit=10.2 -c pytorch
