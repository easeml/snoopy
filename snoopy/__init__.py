import os

from ._cache import set_cache_dir
from ._logging import get_logger


def _init_logging():
    from ._logging import set_logging_level

    ll = "DEBUG"
    try:
        ll = os.environ["SNOOPY_LL"].upper()
        assert ll in ["ALL", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"], f"Unknown logging level: {ll}"

    except KeyError:
        pass

    finally:
        set_logging_level(ll if ll != "ALL" else "DEBUG")

        # Set TF Logging level
        if ll == "ALL":
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        else:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def _init_tf_pytorch():
    import torch as pt
    import tensorflow as tf

    _logger = get_logger(__name__)

    using_all = True

    try:
        limit_cpu = os.environ["SNOOPY_LIMIT_CPU"]
        if "y" in limit_cpu.lower():
            using_all = False
            _logger.info("Limiting CPU usage to 1 core for TF and 1 core for PyTorch")
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            pt.set_num_threads(1)
            pt.set_num_interop_threads(1)

    except KeyError:
        using_all = True

    if using_all:
        _logger.warning("Snoopy may use all CPU cores! Set env. variable 'SNOOPY_LIMIT_CPU' to 'y' to limit CPU usage.")

    # Copied from https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    # This prevents CUBLAS_STATUS_NOT_INITIALIZED masking error CUDA_ERROR_OUT_OF_MEMORY
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            _logger.debug(f"TensorFlow sees: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            _logger.error(e)


# TODO: Check that this does not cause deadlocks!
os.environ["TOKENIZERS_PARALLELISM"] = "true"

_init_logging()
_init_tf_pytorch()
