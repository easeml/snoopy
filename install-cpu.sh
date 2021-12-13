# CPU
conda create --name snoopy-cpu python=3.7
eval "$(conda shell.bash hook)"
conda activate snoopy-cpu
conda install pytorch torchvision cpuonly faiss-cpu -c pytorch
conda install -c conda-forge spacy ftfy matplotlib colorlog scikit-learn
pip install --no-cache-dir tensorflow_cpu tensorflow-addons tfds-nightly tensorflow-hub transformers efficientnet_pytorch
