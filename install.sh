#GPU
conda create --name snoopy python=3.7
conda activate snoopy
conda install -c pytorch pytorch torchvision cudatoolkit=10.1
conda install -c conda-forge spacy ftfy matplotlib colorlog scikit-learn
#TF must be installed via pip, because TF 2.2 is not available via conda
#TF 2.2 is needed, because text cannot be fed to TF Hub models if TF 2.1 is used
pip install --no-cache-dir tensorflow tensorflow-addons tfds-nightly tensorflow-hub transformers efficientnet_pytorch
