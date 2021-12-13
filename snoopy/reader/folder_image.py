# Adapted from: https://www.tensorflow.org/tutorials/load_data/images
import os
from dataclasses import dataclass

import tensorflow as tf

from .base import Reader, ReaderConfig, UNKNOWN_LABEL
from .._logging import get_logger
from ..custom_types import DataType, DataWithInfo

_logger = get_logger(__name__)


# Folder should contain sub-folders whose names are representing labels of images within them
# Images should be JPEG or PNG
# NOTE: Each subfolder is mapped to a label, so make sure to not include irrelevant subfolders!
# Example:
# /path/specified/in/config/
# * first_label/
# * * image1.jpg
# * * image2.jpg
# * second_label/
# * * image3.jpg
# * * image4.jpg
@dataclass(frozen=True)
class FolderImageConfig(ReaderConfig):
    path: str
    num_channels: int

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


class FolderImageReader(Reader):

    @staticmethod
    def read_data(config: FolderImageConfig) -> DataWithInfo:
        # Construct mapping from folder name to an integer
        # # List of all subfolder names
        subfolder_names = sorted(map(lambda x: x.name, filter(lambda x: x.is_dir(), os.scandir(config.path))))
        # # Enumerate each of the subfolders
        mapping = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(subfolder_names),
                                                            values=tf.constant(list(range(len(subfolder_names))))),
            default_value=UNKNOWN_LABEL
        )

        def process_path(file_path):
            parts = tf.strings.split(file_path, os.path.sep)
            label = mapping.lookup(parts[-2])
            img = tf.image.decode_jpeg(tf.io.read_file(file_path), channels=config.num_channels)

            return img, label

        # All files within all sub-folders
        path_wildcard = os.path.join(config.path, "*", "*")

        # Dataset of filenames
        _logger.debug(f"Searching for images in subfolders of {config.path} (found {len(subfolder_names)} subfolders)")
        list_of_files_dataset = tf.data.Dataset.list_files(path_wildcard)
        data_size = tf.data.experimental.cardinality(list_of_files_dataset)
        _logger.debug(f"Found {data_size} images in {len(subfolder_names)} subfolders (labels) of {config.path}")

        # Shuffle dataset before loading images
        list_of_files_dataset = list_of_files_dataset.shuffle(data_size)

        # Load images and labels
        dataset = list_of_files_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Return
        return_value = DataWithInfo(data=dataset, size=data_size.numpy(), num_labels=len(subfolder_names))
        return return_value
