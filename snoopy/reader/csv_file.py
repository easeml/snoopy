# Adapted from: https://www.tensorflow.org/tutorials/load_data/csv
import os
from dataclasses import dataclass
from typing import List, OrderedDict

import tensorflow as tf

from .base import Reader, ReaderConfig, UNKNOWN_LABEL
from .._logging import get_logger
from ..custom_types import DataType, DataWithInfo

_logger = get_logger(__name__)


# CSV file should not contain any whitespace between the delimiter and actual field, in such cases the field should
# be quoted. There should be no empty lines in the CSV file.
@dataclass(frozen=True)
class CSVFileConfig(ReaderConfig):
    path: str
    header_present: bool
    text_column_number: int
    label_column_number: int
    num_columns: int
    label_values: List[str]
    shuffle_buffer_size: int
    delimiter: chr = ','

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT


def _count_file_rows(path: str, header_present: bool) -> int:
    cnt = 0
    with open(path) as f:
        for _ in f:
            cnt += 1

    if header_present and cnt > 0:
        cnt -= 1

    return cnt


class CSVFileReader(Reader):
    @staticmethod
    def read_data(config: CSVFileConfig) -> DataWithInfo:
        # Mapping label -> label index
        mapping = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(config.label_values),
                                                            values=tf.constant(list(range(len(config.label_values))))),
            default_value=UNKNOWN_LABEL
        )

        # index = number - 1
        index_text = config.text_column_number - 1
        index_label = config.label_column_number - 1

        # Define column names so that label column can be identified
        column_names = [str(i) for i in range(config.num_columns)]
        column_names[index_text] = "text"
        column_names[index_label] = "label"

        # Set number of threads to number of available CPU cores or 1 if number cannot be obtained
        cpu_count = os.cpu_count()
        if not cpu_count:
            cpu_count = 1
        _logger.debug(f"{cpu_count} thread(s) will be used to load CSV dataset at {config.path}")

        def get_text_and_numeric_label(line: OrderedDict):
            return line["text"][0], mapping.lookup(line["label"][0])

        data = tf.data.experimental.make_csv_dataset(
            file_pattern=config.path,
            batch_size=1,
            column_names=column_names,
            select_columns=[index_text, index_label],
            field_delim=config.delimiter,
            header=config.header_present,
            num_epochs=1,
            shuffle=True,
            shuffle_buffer_size=config.shuffle_buffer_size,
            num_parallel_reads=cpu_count,
            sloppy=True
        ).map(get_text_and_numeric_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        _logger.debug(f"Counting rows of {config.path}")
        size = _count_file_rows(config.path, config.header_present)
        _logger.debug(f"Counted {size} rows in {config.path}")
        return DataWithInfo(data=data, size=size, num_labels=len(config.label_values))
