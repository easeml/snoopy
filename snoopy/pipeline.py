import os
import threading as th
from collections import OrderedDict as OrdD
from queue import Queue
from time import sleep
from typing import Dict, List, Optional, OrderedDict, Tuple

import numpy as np
import torch as pt

from ._logging import get_logger
from ._utils import get_tf_device
from .custom_types import CacheType, DataType, DataWithInfo, EmbeddingSlice
from .embedding import EmbeddingConfig, EmbeddingDataset, EmbeddingDatasetsTuple, PCASpec
from .reader import ReaderConfig, data_factory
from .result import Observer
from .strategy import StrategyConfig, strategy_factory

_logger = get_logger(__name__)


def _compatibility_check(train: ReaderConfig,
                         test: ReaderConfig,
                         embedding_configs: List[EmbeddingConfig]):
    # Check number of embeddings
    assert len(embedding_configs) > 0, "There should be at least one embedding specified!"

    # Check input data same
    filtered_data_types = [i for i in [train.data_type, test.data_type] if i != DataType.ANY]
    num_distinct_data_types = len(set(filtered_data_types))
    assert num_distinct_data_types <= 1, \
        f"Data type should be same for training and test data! Found '{train.data_type.value}' for train data and " \
        f"'{test.data_type.value}' for test data!"

    # Check embeddings same
    embeddings_types = [embedding_config.embedding_model_spec.data_type for embedding_config in embedding_configs]
    filtered_embedding_types = [i for i in embeddings_types if i != DataType.ANY]
    num_distinct_embedding_types = len(set(filtered_embedding_types))
    assert num_distinct_embedding_types <= 1, "All embeddings should process data of same type!"

    # Check input data same as embeddings
    assert num_distinct_data_types == 0 or num_distinct_embedding_types == 0 or \
           filtered_data_types[0] == filtered_embedding_types[0], \
        f"Specified embeddings process data of type '{filtered_embedding_types[0].value}', whereas the data is of " \
        f"type '{filtered_data_types[0].value}'!"

    # PCA-specific checks
    # Check that there is at most one PCA transformation defined per output dimension
    pca_output_dimensions = [e.embedding_model_spec.output_dimension for e in embedding_configs if
                             isinstance(e.embedding_model_spec, PCASpec)]

    assert len(set(pca_output_dimensions)) == len(pca_output_dimensions), \
        "There can be at most one PCA transformation defined per unique PCA output dimension!"


def run(train_data_config: ReaderConfig,
        test_data_config: ReaderConfig,
        embedding_configs: OrderedDict[str, EmbeddingConfig],
        strategy_config: StrategyConfig,
        observer: Observer,
        device: pt.device):
    # Check that data and embeddings are compatible
    _compatibility_check(train_data_config, test_data_config, list(embedding_configs.values()))

    # Prepare raw data
    train_data_raw = data_factory(train_data_config)
    test_data_raw = data_factory(test_data_config)

    # Transform config Objects into EmbeddingDataset
    embedding_datasets = OrdD()

    for key in embedding_configs:
        # PCA is a special case, it is run immediately on the CPU, strategy has only one lever pull available
        # Special behaviour is needed, because the transformation depends on the whole training dataset
        if isinstance(embedding_configs[key].embedding_model_spec, PCASpec):
            min_batch_size = max(train_data_raw.size, test_data_raw.size)
            assert embedding_configs[key].batch_size >= min_batch_size, \
                f"For PCA embedding, batch size must be large to fit whole training/test set. " \
                f"For the dataset specified it has to be at least {min_batch_size}!"

            # Prepare embedding
            train = EmbeddingDataset(train_data_raw, embedding_configs[key], CacheType.DEVICE, device)
            test = EmbeddingDataset(test_data_raw, embedding_configs[key], CacheType.DEVICE, device)

            # IMPORTANT: order of calls defines what is used for training and what for testing
            train.prepare()
            test.prepare()

        else:
            train = EmbeddingDataset(train_data_raw, embedding_configs[key], CacheType.NONE, device)
            test = EmbeddingDataset(test_data_raw, embedding_configs[key], CacheType.DEVICE, device)

        embedding_datasets[key] = EmbeddingDatasetsTuple(train=train, test=test)

    # Initialize strategy
    strategy = strategy_factory(strategy_config)

    # Run strategy
    strategy.execute(datasets=embedding_datasets, observer=observer)


def store_embeddings(train_data_config: ReaderConfig,
                     test_data_config: ReaderConfig,
                     embedding_configs: OrderedDict[str, EmbeddingConfig],
                     output_files_path: str,
                     device: Optional[pt.device] = None,
                     filename_mapping: Dict[str, str] = None):
    # Use name given to embedding as filename
    if filename_mapping is None:
        filename_mapping = {x: x for x in embedding_configs}

    assert set(embedding_configs.keys()) == set(filename_mapping.keys()), \
        f"Filename should be specified for each embedding!"

    # Check that data and embeddings are compatible
    _compatibility_check(train_data_config, test_data_config, list(embedding_configs.values()))

    # Prepare raw data
    train_data_raw = data_factory(train_data_config)
    test_data_raw = data_factory(test_data_config)

    # Parallel execution variables
    # #  Which devices will be used for execution
    # # # 1. If device is specified, use only that device
    if device:
        num_devices = 1
        available_devices = [device]

    # # # 2. If device is not specified use all available GPUs
    else:
        num_devices = pt.cuda.device_count()
        assert num_devices > 0, "There are no GPU devices available!"
        available_devices = [pt.device("cuda", i) for i in range(num_devices)]

    _logger.info(f"{num_devices} device(s) will be used for running inference:"
                 f" {', '.join(map(get_tf_device, available_devices))}")

    # # Used to signal result of embedding inference to the main thread
    queue = Queue()

    # # One flag per device signalling whether a device is free to use
    is_worker_free = [True] * num_devices

    # # Counters (indices) of jobs started and finished
    index_started = 0
    index_finished = 0

    # # Mapping that is used to signal that a device is free after a job is done executing
    mapping_key_to_worker_index = {}

    # # Info about embeddings
    num_embeddings = len(embedding_configs)
    keys_list = list(embedding_configs.keys())

    # Used by job function
    def job_inner(data_raw: DataWithInfo, config: EmbeddingConfig, target_device: pt.device) -> \
            Tuple[np.ndarray, np.ndarray]:
        ed = EmbeddingDataset(data_raw, config, CacheType.CPU, target_device)
        ed.prepare()
        es: EmbeddingSlice = ed.get_cache(0, ed.size, copy_to_device=False)
        del ed

        # Since copy_to_device is set to False and CPU cache is used, there is no need to transfer data to CPU
        features = es.features.numpy()
        labels = es.labels.numpy()

        return features, labels

    # Run inference of embedding on specified device and put result to queue
    def job(key: str, config: EmbeddingConfig, target_device: pt.device):
        # Uses variables defined in function: train_data_raw, test_data_raw, queue

        # Training data
        _logger.info(f"Computing '{key}' embeddings for train dataset on {get_tf_device(target_device)}")
        train_features, train_labels = job_inner(train_data_raw, config, target_device)

        # Test data
        _logger.info(f"Computing '{key}' embeddings for test dataset on {get_tf_device(target_device)}")
        test_features, test_labels = job_inner(test_data_raw, config, target_device)

        queue.put((key, (train_features, train_labels, test_features, test_labels)))

    # We are done when all jobs are finished
    while index_finished < num_embeddings:
        # Check whether there is a job to start or finish every second
        sleep(1)

        # Start new jobs if possible
        while index_started < num_embeddings and True in is_worker_free:
            index_free_worker = is_worker_free.index(True)
            is_worker_free[index_free_worker] = False

            # Get key
            key_ = keys_list[index_started]

            # Get device used for executing a job
            job_device = available_devices[index_free_worker]

            # Store worker index, so that worker can be marked as free after the job completes
            mapping_key_to_worker_index[key_] = index_free_worker

            # Handle PCA as a special case
            if isinstance(embedding_configs[key_].embedding_model_spec, PCASpec):
                min_batch_size = max(train_data_raw.size, test_data_raw.size)
                assert job_device.type == "cpu", \
                    "PCA can only be executed on CPU! Specify device as CPU instead to ensure execution on CPU!"

                assert embedding_configs[key_].batch_size >= min_batch_size, \
                    f"For PCA embedding, batch size must be large to fit whole training/test set. " \
                    f"For the dataset specified it has to be at least {min_batch_size}!"

            t = th.Thread(target=job, args=(key_, embedding_configs[key_], job_device))
            t.start()

            # Signal that this job was launched
            index_started += 1

        # Handle finished jobs
        while not queue.empty():
            key_, data = queue.get()

            # Mark worker as available
            index_completed_worker = mapping_key_to_worker_index[key_]
            is_worker_free[index_completed_worker] = True

            # Signal that job was finished
            index_finished += 1

            # Store processed data
            _logger.info(f"Storing '{key_}' embeddings")
            np.savez(os.path.join(output_files_path, filename_mapping[key_]),
                     train_features=data[0],
                     train_labels=data[1],
                     test_features=data[2],
                     test_labels=data[3])
