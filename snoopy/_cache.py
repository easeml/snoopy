import os
from typing import Optional

import torch as pt

_path = ""


def set_cache_dir(path: str) -> None:
    global _path
    _path = path

    # TensorFlow
    os.environ["TFHUB_CACHE_DIR"] = os.path.join(path, "TensorFlow_cache")

    # PyTorch
    pt.hub.set_dir(os.path.join(path, "PyTorch_cache"))


def get_hugging_face_cache_dir() -> Optional[str]:
    global _path
    if _path == "":
        return None
    else:
        return os.path.join(_path, "HuggingFace_cache")


def get_tfds_cache_dir() -> Optional[str]:
    global _path
    if _path == "":
        return None
    else:
        return os.path.join(_path, "TFDS_cache")
