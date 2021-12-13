from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import torch as pt
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast

from .base import EmbeddingModel, EmbeddingModelSpec
from .._cache import get_hugging_face_cache_dir
from .._logging import get_logger
from .._utils import get_tf_device
from ..custom_types import DataType

_logger = get_logger(__name__)


@dataclass
class HuggingFaceSpec(EmbeddingModelSpec):
    name: str
    output_dimension: int
    max_length: int
    fast_tokenizer: bool = False
    tokenizer_params: dict = None
    decode_format: str = "utf-8"

    def load(self, device: pt.device) -> EmbeddingModel:
        return _Inner.get_instance(self, device)

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT


class _Inner(EmbeddingModel):
    _instances = dict()

    def __init__(self, spec: HuggingFaceSpec, device: pt.device):
        # Prepare tokenizer
        if spec.tokenizer_params is None:
            tokenizer_params = {}
        else:
            tokenizer_params = spec.tokenizer_params

        # Fast tokenizers (implemented in huggingface/tokenizers) cannot be obtained from AutoTokenizer
        # Instead, each has to be handled separately
        if spec.fast_tokenizer:
            if "bert" in spec.name.lower():
                self._tokenizer = BertTokenizerFast.from_pretrained(spec.name, cache_dir=get_hugging_face_cache_dir(),
                                                                    **tokenizer_params)
                _logger.info(f"Using fast tokenization for {spec.name}")
            else:
                raise RuntimeError(f"Fast tokenization for {spec.name} is not implemented!")

        else:
            self._tokenizer = AutoTokenizer.from_pretrained(spec.name, cache_dir=get_hugging_face_cache_dir(),
                                                            **tokenizer_params)

        # Prepare model
        model = AutoModel.from_pretrained(spec.name, cache_dir=get_hugging_face_cache_dir())
        model.eval()
        model.to(device)
        self._model = model

        # Other params
        self._output_dimension = spec.output_dimension
        self._device = device
        self._max_length = spec.max_length
        self._decode_format = spec.decode_format

    @classmethod
    def get_instance(cls, spec: HuggingFaceSpec, device: pt.device) -> EmbeddingModel:
        combination_string = (spec.name, get_tf_device(device))
        if combination_string not in cls._instances:
            _logger.info(f"Initializing {combination_string[0]} on {combination_string[1]}")
            cls._instances[combination_string] = cls(spec, device)

        return cls._instances[combination_string]

    def move_to(self, device: pt.device) -> None:
        self._model.to(device)
        self._device = device

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    def get_data_preparation_function(self) -> Callable:
        def fn(feature: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            tokenizer_dict = self._tokenizer.encode_plus(
                feature.numpy().decode(self._decode_format),  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self._max_length,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_token_type_ids=True,
                return_tensors="tf")

            return tokenizer_dict["input_ids"], tokenizer_dict["attention_mask"], tokenizer_dict["token_type_ids"]

        return lambda feature: tf.py_function(func=fn, inp=[feature], Tout=(tf.int32, tf.int32, tf.int32))

    # Inspired by: https://discuss.pytorch.org/t/select-data-through-a-mask/45598/4
    @staticmethod
    @pt.jit.script
    def _get_mean_of_relevant_tokens(all_: pt.Tensor, attention_mask: pt.Tensor) -> pt.Tensor:
        # #inputs x 1
        count_relevant_tokens = pt.sum(attention_mask, dim=1, keepdim=True)

        # Prevent division by 0
        # Why pt.tensor? https://discuss.pytorch.org/t/torchscript-indexing-question-filling-nans/53100
        count_relevant_tokens[count_relevant_tokens == 0] = pt.tensor(1, dtype=count_relevant_tokens.dtype)

        # #inputs x #tokens x output_dimension
        all_w_irrelevant_set_to_zero = pt.mul(all_.permute(2, 0, 1), attention_mask).permute(1, 2, 0)

        # Manually compute mean
        return pt.div(pt.sum(all_w_irrelevant_set_to_zero, dim=1), count_relevant_tokens)

    def apply_embedding(self, features: np.ndarray) -> pt.Tensor:
        features = features.squeeze(axis=2)

        features_pt = pt.as_tensor(features, dtype=pt.long, device=self._device)
        input_ids = features_pt[:, 0, :]
        attention_mask = features_pt[:, 1, :]
        token_type_ids = features_pt[:, 2, :]

        with pt.no_grad():
            # Dimensions: #inputs x #tokens x output dimension
            all_embeddings = self._model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)[0]

        return self._get_mean_of_relevant_tokens(all_embeddings, attention_mask)
