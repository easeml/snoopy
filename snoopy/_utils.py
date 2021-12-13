import torch as pt


def get_num_splits(size: int, split_size: int) -> int:
    if size % split_size == 0:
        return size // split_size

    return (size + split_size - 1) // split_size


def get_tf_device(device: pt.device):
    if device.type == "cuda":
        name = "GPU"
        index = device.index
        if index is None:
            index = 0
    elif device.type == "cpu":
        name = "CPU"
        index = 0
    else:
        raise RuntimeError(f"Unknown device type: {device.type}!")

    return f"{name}:{index}"
