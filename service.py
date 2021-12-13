from typing import Optional, List

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

import base64
from io import BytesIO
import os, os.path
import time
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict

import torch as pt
from tensorflow_datasets import Split

from snoopy import set_cache_dir
from snoopy.embedding import EmbeddingConfig, googlenet, alexnet, ImageReshapeSpec, inception, vgg19, efficientnet_b7, uni_se, bert
from snoopy.pipeline import run, store_embeddings
from snoopy.reader import FolderImageConfig, TFDSImageConfig, TFDSTextConfig
from snoopy.result import DemoResultStoringObserver
from snoopy.strategy import SimpleStrategyConfig
from snoopy.reader import ReaderConfig, data_factory

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

dataset_raw_img_size = {
        "mnist": (28, 28),
        "cifar10": (32, 32),
        "cifar100": (32, 32),
        }

dataset_raw_img_channels = {
        "mnist": 1,
        "cifar10": 3, 
        "cifar100": 3, 
        }

dataset_classes = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "imdb_reviews": 2,
        }

dataset_type = {
        "mnist": "IMAGE",
        "cifar10": "IMAGE",
        "cifar100": "IMAGE",
        "imdb_reviews": "TEXT",
        }

folder = "."
device = "cpu"
#device = "cuda:0"


cache_dir = folder + "/cache"
result_dir = folder + "/results"


progress_dir = folder
progress_ending = ".prog"

def get_model_and_filenames(dataset : str, label_noise: Optional[float] = None):
    if dataset_type[dataset] == "IMAGE":
        models = OrderedDict({
                "dummy_img": EmbeddingConfig(ImageReshapeSpec(dataset_raw_img_size[dataset], dataset_raw_img_channels[dataset]), batch_size=10, prefetch_size=1, label_noise_amount = label_noise),
                "googlenet": EmbeddingConfig(googlenet, batch_size=10, prefetch_size=1, label_noise_amount = label_noise),
                "alexnet": EmbeddingConfig(alexnet, batch_size=10, prefetch_size=1, label_noise_amount = label_noise),
                "inceptionv3": EmbeddingConfig(inception, batch_size=10, prefetch_size=1, label_noise_amount = label_noise),
                "vgg19": EmbeddingConfig(vgg19, batch_size=10, prefetch_size=1, label_noise_amount = label_noise),
                "efficientnetb7": EmbeddingConfig(efficientnet_b7, batch_size=10, prefetch_size=1, label_noise_amount = label_noise),
            })
    else:
        models = OrderedDict({
                "use": EmbeddingConfig(uni_se, batch_size=10, prefetch_size=1, label_noise_amount = label_noise),
                "bert": EmbeddingConfig(bert, batch_size=10, prefetch_size=1, label_noise_amount = label_noise),
            })
    if label_noise is None:
        filename_mapping = {name: dataset+ "-" + name for name in models}
    else:
        filename_mapping = {name: dataset+ "-LabelNoise" + str(int(label_noise*100)) + "-" + name for name in models}
    return models, filename_mapping

labels = {
    "dummy_img": "Raw",
    "googlenet": "GoogleNet",
    "alexnet": "AlexNet",
    "inceptionv3": "InceptionV3",
    "vgg19": "VGG-19",
    "efficientnetb7": "EfficientNet-B7",

    "use": "Universal Sentence Encoder (USE)",
    "bert": "BERT",
}

color_values = {
    "dummy_img": "Black",
    "googlenet": "Red",
    "alexnet": "Green",
    "inceptionv3": "Orange",
    "vgg19": "Cyan",
    "efficientnetb7": "Magenta",

    "use": "Red",
    "bert": "Green",
}

def run_pipeline(dataset, missing_models, filename_mapping):
    set_cache_dir(cache_dir)

    train_data = TFDSImageConfig(dataset_name=dataset, split=Split.TRAIN) if dataset_type[dataset] == "IMAGE" else TFDSTextConfig(dataset_name=dataset, split=Split.TRAIN)
    test_data = TFDSImageConfig(dataset_name=dataset, split=Split.TEST) if dataset_type[dataset] == "IMAGE" else TFDSTextConfig(dataset_name=dataset, split=Split.TEST)

    train_data_raw = data_factory(train_data)

    observer = DemoResultStoringObserver(result_dir, progress_dir, progress_ending, train_data_raw.size, filename_mapping)

    run(train_data_config=train_data,
        test_data_config=test_data,
        embedding_configs=missing_models,
        strategy_config=SimpleStrategyConfig(train_size=100, test_size=5_000),
        observer=observer,
        device=pt.device(device))

    observer.store(result_dir, filename_mapping)

    for k in missing_models.keys():
        path = os.path.join(progress_dir, filename_mapping[k] + progress_ending)
        # Update the progress file
        with open(path, "w") as f:
            f.write("Done")

@app.get("/")
def read_root():
    return "Snoopy REST API up and running!"

@app.put("/put")
async def put(background_tasks: BackgroundTasks, dataset : str, embeddings : Optional[List[str]] = Query(None), label_noise: Optional[float] = None):

    models, filename_mapping = get_model_and_filenames(dataset, label_noise)

    if embeddings is None:
        return "No job had to be created!"
    assert len([e for e in embeddings if e not in filename_mapping.keys()]) == 0, "Some embeddings were not found!"

    models_missing = OrderedDict()

    run = False
    for k, v in filename_mapping.items():
        if k not in embeddings:
            continue
        path = os.path.join(progress_dir, v + progress_ending)
        if not os.path.exists(path):
            run = True
            with open(path, "w") as f:
                f.write("Pending")
            models_missing[k] = models[k]

    if not run:
        return "No job had to be created!"

    background_tasks.add_task(run_pipeline, dataset, models_missing, filename_mapping)
    return "Job for {} modules is created in the background.".format(len(models_missing))

@app.get("/check")
def check(dataset : str, embeddings : Optional[List[str]] = Query(None), label_noise: Optional[float] = None):
    
    models, filename_mapping = get_model_and_filenames(dataset, label_noise)

    if embeddings is None:
        return "True"
    assert len([e for e in embeddings if e not in filename_mapping.keys()]) == 0, "Some embeddings were not found!"


    for k, v in filename_mapping.items():
        if k in embeddings:
            path = os.path.join(progress_dir, v + progress_ending)
            if not os.path.exists(path):
                return False
        
    return True

def _get_lowerbound(value, classes):
    return ((classes - 1.0)/float(classes)) * (1.0 - math.sqrt(max(0.0, 1 - ((float(classes) / (classes - 1.0)) * value))))

def create_plot(dataset, embeddings, train_numbers, errors, target = None, train_size = 0):

    assert len(embeddings) == len(train_numbers) and len(embeddings) == len(errors)

    sns.set(font="lato")
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)

    min_tn = 1.0
    f = plt.figure(figsize=(7, 5))
    for idx, emb in enumerate(embeddings):

        ax = sns.lineplot(train_numbers[idx], errors[idx], color=color_values[emb], label=labels[emb])

        min_tn = min(min_tn, min(errors[idx]))

        ax.lines[-1].set_linestyle("--")
        #ax.lines[-1].set_label("{}".format(labels[emb]))

    if target is not None:
      x = [0, train_size]
      sns.lineplot(x, [target]*len(x), ax=ax)
      ax.lines[-1].set_color("Blue")
      ax.lines[-1].set_linestyle("--")
      ax.lines[-1].set_label("Target Error")

    ax.legend()

    ax.set_ylim(0.0, min(1.0, max(target, min_tn)*2.0))
    ax.set_ylabel("BER Estimation")
    ax.set_xlabel("Train Samples")

    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title

    buf = BytesIO()
    f.savefig(buf, bbox_inches = 'tight', pad_inches = 0.2, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    #return img
    return f"<img src='data:image/png;base64,{data}'/>"

@app.get("/get")
def get(target: float, dataset : str, embeddings : Optional[List[str]] = Query(None), label_noise: Optional[float] = None):

    models, filename_mapping = get_model_and_filenames(dataset, label_noise)

    assert target >= 0.0 and target <= 1.0
    target = 1.0 - target

    if embeddings is None:
        return "Pending", "", ""
    assert len([e for e in embeddings if e not in filename_mapping.keys()]) == 0, "Some embeddings were not found!"

    set_cache_dir(cache_dir)
    test_data = TFDSImageConfig(dataset_name=dataset, split=Split.TEST) if dataset_type[dataset] == "IMAGE" else TFDSTextConfig(dataset_name=dataset, split=Split.TEST)
    test_data_raw = data_factory(test_data)
    test_size = float(test_data_raw.size)
    train_data = TFDSImageConfig(dataset_name=dataset, split=Split.TRAIN) if dataset_type[dataset] == "IMAGE" else TFDSTextConfig(dataset_name=dataset, split=Split.TRAIN)
    train_data_raw = data_factory(train_data)
    train_size = float(train_data_raw.size)

    done = True
    res = {}
    es = []
    ns = []
    errs = []
    success = False
    for k, v in filename_mapping.items():
        if k not in embeddings:
            continue
        path = os.path.join(progress_dir, v + progress_ending)
        if not os.path.exists(path):
            res[k] = "Missing"
            done = False
        else:
            with open(path, "r") as f:
                res[k] = f.readline()
                if not res[k].startswith("Pending"):
                    npz_path = os.path.join(result_dir, v + ".npz")
                    data = np.load(npz_path)
                    n = data["n"]
                    err = [_get_lowerbound(x/test_size, dataset_classes[dataset]) for x in data["err"]]
                    if min(err) <= target:
                        success = True
                    es.append(k)
                    ns.append(n)
                    errs.append(err)
                if not res[k].startswith("Done"):
                    done = False

    if len(es) == 0:
        return "Pending", res, ""

    # Create graph and return the image
    image = create_plot(dataset, es, ns, errs, target, train_size)
    overall_state = "Achievable" if success else "NotAchievable" if done else "Running"

    return overall_state, res, image
