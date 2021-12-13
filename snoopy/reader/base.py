from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..custom_types import DataType, DataWithInfo

UNKNOWN_LABEL = -1


@dataclass(frozen=True)
class ReaderConfig:

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        pass


# Strategy pattern, Reader here is behaviour
class Reader(ABC):

    @staticmethod
    @abstractmethod
    def read_data(config: ReaderConfig) -> DataWithInfo:
        pass
