---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# CLIP Benchmark

* This notebook measures the performance of CLIP Image Encoder on Imagenetv2 dataset using ```Pytorch``` and ```Tensorflow```

```{code-cell} ipython3

```

### PyTorch CLIP Model

* With ```batch_size=64```, model takes ```54 seconds``` to process ```~10k``` images.

```{code-cell} ipython3
import numpy as np
import torch
import clip
from tqdm.notebook import tqdm

# Load Model and preprocess for images
model, preprocess = clip.load("ViT-B/32")
device = torch.device('cuda')
model.to(device)
model.eval()

! pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

from imagenetv2_pytorch import ImageNetV2Dataset
images = ImageNetV2Dataset(transform=preprocess)
loader = torch.utils.data.DataLoader(images, batch_size=64, num_workers=2)

all_data = []
with torch.no_grad():
    for i, (images, target) in enumerate(tqdm(loader)):
        images = images.cuda()
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
```

```{code-cell} ipython3

```

### Load CLIP Model tf-transformers

```{code-cell} ipython3
from tf_transformers.models.clip import CLIPModel, CLIPFeatureExtractorTF
from transformers import CLIPTokenizer

import tensorflow as tf
import tqdm

# Model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", return_layer=True)
# Get text and image encoder out
text_encoder = model.text_encoder
image_encoder = model.image_encoder
```

```{code-cell} ipython3

```

### TF with clip preprocess (pt to tf data)

* With ```batch_size=64```, model takes ```54 seconds``` to process ```~10k``` images.

```{code-cell} ipython3
images = ImageNetV2Dataset(transform=preprocess)
loader = torch.utils.data.DataLoader(images, batch_size=64, num_workers=2)

for i, (images, target) in tqdm.tqdm(enumerate(loader)):
    images = {'input_pixels': tf.transpose(tf.convert_to_tensor(images.numpy()), [0, 2, 3, 1] )}
    outputs = image_encoder(images)
```

### TF with tf.io preprocess (Preprocess on the fly)

* With ```batch_size=64```, model takes ```17 seconds``` to process ```~10k``` images. This is ```3x``` times faster.

```{code-cell} ipython3
img_height = 224
img_width = 224
rescaler = tf.keras.layers.Rescaling(scale=1.0/255.0)
mean = [0.48145466, 0.4578275, 0.40821073]
variance = [0.26862954, 0.26130258, 0.27577711]

def standardize(image_data):
    image_data -= tf.constant([mean])
    image_data /= tf.constant([variance])
    return image_data

def read_process_resize(image_path: str):
    """Read, decode and process"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    result = {}
    result['image_path'] = image_path
    result['input_pixels'] = standardize(rescaler(img))
    result['label'] = tf.strings.split(image_path, '/')[2] # string
    return result

image_files = tf.constant(tf.io.gfile.glob("Imagenet/imagenetv2-matched-frequency-format-val/*/*.jpeg"))
image_dataset = tf.data.Dataset.from_tensor_slices(image_files)
image_dataset = image_dataset.map(read_process_resize, num_parallel_calls=tf.data.AUTOTUNE)
batch_size = 64
image_dataset = image_dataset.batch(batch_size, drop_remainder=False)

for (index, item) in tqdm.tqdm(enumerate(image_dataset)):
    image_features = image_encoder(item)['cls_output']
    image_features = tf.nn.l2_normalize(image_features, axis=-1)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
