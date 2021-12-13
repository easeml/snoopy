from dataclasses import dataclass
from typing import Union

import tensorflow_datasets as tfds

from .base import Reader, ReaderConfig
from .._cache import get_tfds_cache_dir
from .._logging import get_logger
from ..custom_types import DataType, DataWithInfo

_logger = get_logger(__name__)


# TODO: shuffle size should be a parameter
@dataclass(frozen=True)
class TFDSImageConfig(ReaderConfig):
    dataset_name: str
    split: tfds.Split

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


@dataclass(frozen=True)
class TFDSTextConfig(ReaderConfig):
    dataset_name: str
    split: tfds.Split

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT


class TFDSReader(Reader):
    @staticmethod
    def read_data(config: Union[TFDSImageConfig, TFDSTextConfig]) -> DataWithInfo:
        info: tfds.core.DatasetInfo
        data, info = tfds.load(config.dataset_name, split=config.split, data_dir=get_tfds_cache_dir(),
                               shuffle_files=True, with_info=True, as_supervised=True)

        data_size = info.splits[config.split].num_examples
        num_labels = info.features["label"].num_classes
        return_value = DataWithInfo(data=data.shuffle(data_size), size=data_size, num_labels=num_labels)
        _logger.debug(f"Loaded text dataset {info.name} (split: {config.split}) with {data_size} points")
        return return_value
