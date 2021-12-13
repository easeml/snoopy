from .base import EmbeddingConfig, EmbeddingDataset, EmbeddingDatasetsTuple, EmbeddingIterator, EmbeddingModelSpec
from .dummy import DummySpec, ImageReshapeSpec
from .hugging_face import HuggingFaceSpec
from .sklearn_transform import PCASpec
from .tf_hub import TFHubImageSpec, TFHubTextSpec
from .torch_hub import TorchHubImageSpec

# Image embeddings
inception = TFHubImageSpec(url="https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
                           output_dimension=2048, required_image_size=(299, 299))

mobilenet = TorchHubImageSpec(name="mobilenet_v2", output_dimension=1280,
                              layer_extractor=lambda x: x.classifier[0], required_image_size=(224, 224))

vgg19 = TorchHubImageSpec(name="vgg19", output_dimension=4096, layer_extractor=lambda x: x.classifier[5],
                          required_image_size=(224, 224))

resnet_152 = TorchHubImageSpec(name="resnet152", output_dimension=2048, layer_extractor=lambda x: x.avgpool,
                               required_image_size=(224, 224))

alexnet = TorchHubImageSpec(name="alexnet", output_dimension=4096, layer_extractor=lambda x: x.classifier[5],
                            required_image_size=(224, 224))

googlenet = TorchHubImageSpec(name="googlenet", output_dimension=1024, layer_extractor=lambda x: x.dropout,
                              required_image_size=(224, 224))

efficientnet_b7 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
                                 output_dimension=2560, required_image_size=(600, 600))

efficientnet_b7_torch = TorchHubImageSpec(name="efficientnet-b7", output_dimension=2560,
                                          layer_extractor=lambda x: x._dropout, required_image_size=(600, 600))

# Text embeddings
bert = HuggingFaceSpec(name="bert-base-uncased", output_dimension=768, max_length=512, fast_tokenizer=True)

openai_gpt = HuggingFaceSpec(name="openai-gpt", output_dimension=768, max_length=512,
                             tokenizer_params={"pad_token": "<pad>"})

xlnet = HuggingFaceSpec(name="xlnet-base-cased", output_dimension=768, max_length=512)

uni_se = TFHubTextSpec(url="https://tfhub.dev/google/universal-sentence-encoder/4", output_dimension=512)

nnlm = TFHubTextSpec(url="https://tfhub.dev/google/nnlm-en-dim128/2", output_dimension=128)
