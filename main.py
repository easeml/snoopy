from collections import OrderedDict

from absl import logging

import math
import numpy as np
import os.path as path
import torch as pt
from tensorflow_datasets import Split

from snoopy import set_cache_dir
from snoopy.embedding import EmbeddingConfig, googlenet
from snoopy.pipeline import run, store_embeddings
from snoopy.reader import FolderImageConfig, TFDSImageConfig
from snoopy.result import ResultStoringObserver
from snoopy.strategy import SimpleStrategyConfig

if __name__ == "__main__":
    set_cache_dir("cache")

    #dataset_name = "mnist"
    dataset_name = "cifar10"
    train_data = TFDSImageConfig(dataset_name=dataset_name, split=Split.TRAIN)
    test_data = TFDSImageConfig(dataset_name=dataset_name, split=Split.TEST)
    classes = 10

    # train_data = FolderImageConfig(path="/home/rengglic/ImageNet/train", num_channels=3)
    # test_data = FolderImageConfig(path="/home/rengglic/ImageNet/val", num_channels=3)
    # classes = 1000

    models = OrderedDict({
        "googlenet": EmbeddingConfig(googlenet, batch_size=100, prefetch_size=1)
    })  

    observer = ResultStoringObserver()

    run(train_data_config=train_data,
        test_data_config=test_data,
        embedding_configs=models,
        strategy_config=SimpleStrategyConfig(train_size=100, test_size=5_000),
        observer=observer,
        device=pt.device("cpu"))

    folder = "results"
    observer.store(folder)
    
    def _get_lowerbound(value, classes):
        return ((classes - 1.0)/float(classes)) * (1.0 - math.sqrt(max(0.0, 1 - ((float(classes) / (classes - 1.0)) * value))))

    min_error = (classes - 1.0) / float(classes)
    for k in models.keys():
        f = path.join(folder, "{0}.npz".format(k))
        if path.exists(f):
            items = np.load(f)
            error = items['err'][-1] / float(items['n'][-1])
            min_error = min(min_error, error)
    
    logging.info('Minimal error achievable is %4f', _get_lowerbound(min_error, classes))
