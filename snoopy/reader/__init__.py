from .base import Reader, ReaderConfig, UNKNOWN_LABEL
from .csv_file import CSVFileConfig, CSVFileReader
from .folder_image import FolderImageConfig, FolderImageReader
from .tfds import TFDSImageConfig, TFDSReader, TFDSTextConfig
from ..custom_types import DataWithInfo


def data_factory(reader_config: ReaderConfig) -> DataWithInfo:
    if isinstance(reader_config, TFDSImageConfig):
        return TFDSReader.read_data(reader_config)

    elif isinstance(reader_config, TFDSTextConfig):
        return TFDSReader.read_data(reader_config)

    elif isinstance(reader_config, FolderImageConfig):
        return FolderImageReader.read_data(reader_config)

    elif isinstance(reader_config, CSVFileConfig):
        return CSVFileReader.read_data(reader_config)

    else:
        raise NotImplementedError
