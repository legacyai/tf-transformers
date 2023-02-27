---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:id: oQJ4kzmr0QFb


```

+++ {"id": "sDAuhGSA0dze"}

# Read and Write Images as TFRecords

Tensorflow-Transformers makes it easy to read and write tfrecords in and from any data type.
Here we will see how we can make use of it to write and read images as tfrecords

```{code-cell} ipython3
:id: c4x5NBTg0eaW


```

```{code-cell} ipython3
:id: 4SYpRIzc0fxm

import json
import glob
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tf_transformers.data import TFWriter, TFReader
from datasets import load_dataset
```

```{code-cell} ipython3
:id: cqx86-bB0gNh


```

+++ {"id": "E-vLJ5M9003N"}

## Load CelebA dataset from HF

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 315
  referenced_widgets: [836376a41735471dbf95b16b08342d8d, 51f45e88fd66414587501b0dafbbdc89,
    e586b23e00fd4a3d9bc53737e6284bc2, d9ca2afedc04491aa7b7677df89117ff, 452037daee604d719b28367e9a1f7110,
    c632f10b38ac45ee86f47f7542e0ce46, 4d7563a5397e4237ada04f6bca9d2319, 84e45737df9c43d192379240bb4c14fc,
    6569e36ddac84f6c962cd6707d593f0f, 34472bd117e0412da284837c5041c568, 8c8338d4473a4f2ca1405657be98ace8,
    c7cd0b4b47ad4bae8500bbf584434d3a, 6d280a7e45674ba4a98423f763d763bf, cfe43b6f59ae4cf492595cd70577f8c0,
    58bfa0430449416797ea5dd69f1c0f14, b655bae8189f444eb0ce8b43ed1a8295, 86c1a3dfa8994f7e81ba92f15f7a9ead,
    07ea3e4ee2a9407f9f020a7842eb4a6c, e985bb632072405f9a751b28e188cf23, 4f924c32e4a54285aaedf365234ef14e,
    76304919b9b442f8ba50a712c628c29f, f58561e6599b4070b35b85676c2fdbfc, 762b19366aa145c19716861105edef64,
    4d4066daf83848ab8c0fee96314d13ea, 85b954c48a084b9084ab3ecd770a16e5, 740e995f84e745dea02d7eed5626b3c4,
    0b66c484e1364b1a8baf728e5874b5d5, b417e16ebdd44215b9b73d0a7a129276, fe10a9676233483b871c315d8e872035,
    7ce489f589f4412c99902f97a3f8859c, f31e6948e0c44c6398898369e07e713f, a26c4fbe69174069ab98b57600e76ee3,
    36061e6469d24e0fb21cab5ff0310d11, 275a67c3aec34aecac01ef88fb0d6e42, c2f978c6f947409c9bd66b55303b600d,
    cadd19345a6341ce92441ba8f8b8262e, f8c8e2342ea94769a4312e4268cd127e, b38c9fea09364934a5af9bb07bedad1b,
    8f84e971a86044468d29b388b6e31a6c, 38d22f1f07f84810b13a6b588120bb74, 52230587cff74e27a31b6a2dc14e4c6c,
    b8b7473fd14041c88d67e72c490193ad, 499c5c6a217547069eb20e7796645df6, 6026392b2f154af58868232d52bad25f,
    7f7b841fe2a44ac890b9cfa6c411756d, 18b094566c474bc19ee2d076a0238066, 04f57d2225614db9aa81748c22cfa7b5,
    c4a3a703ce2d403db7b1e92fbe0a1874, 818d8d0f4bc6486bae0a8a3dcd8d1622, 7b4a70133afa44059356b42f1b7acc81,
    49d189e9f4034288a763d52d00470073, 04c3ae6f4e7e4acb94eb7bf927acdaef, a4f0ece850ba46d0ad562ab44ec108bc,
    8093c58c12104435ad67e54c527c5894, 54494243505246b8ab2dbb01d3e13b1c, b1f283f2340149b9aa9fb2296eff3312,
    df9c3fdcf08c460ab272d63eac605a70, afc707e0144e4625bc188f284a3c6923, f4d59f4ac71b4165a47af633c54e65fc,
    5962e1d214374e7ab529545d397ed689, a3e912ac14f44e6eb9cace7c073c097f, 1cbb3ff9f07e4224afdd32d0c8d0878a,
    566592c616144ea288e1994a72af5477, bfef7a88d0fc47018fe426c898ce359e, 120b1fd6be3e43fbbe62aa7362082cb6,
    05f8eb9c34fa46adb6506fad2204f9fc, a3bd6d36cf1445a4866b2af47cae0821, 16901721fb394a8388bb45c47234dd9d,
    d6555447fe414edbbd62b58a9f54385a, 2b3247123e984feb995f79026bff8d14, ffe3f8703fe04a00871c83a39735ac86,
    79f317abafcf4b0fae7ada867ea64a9f, caf1938cf18f4ee08ec48c36e82c33a8, 1049b4f6772e4049aaa1d89e54cf5bda,
    f0c0e1a93c234b16aac03e54adbc4b26, be68f84c4f4f404681079e6cf1a82cef, 1682d04ba34d4c73a339df4e3d25dd04,
    ab1a055c23e94d28b61c331a8e8759f6, 960800bdd14f4f9bac78cbe1c93f9445, c31d5105071243799eaff0794d829019,
    3665df09187d4785b6cb5ca2da211e76, dc65ec87091a47939515c5c453495dc2, 27148daa533b4475b1fabd39b87d669d,
    bc0429d647664aa79a2f09fac8b890b3, a5d3958c163a480bbbacdce4d52500b4, b8e082a7de6c4b0f8be797a93bdffa90,
    a388c5de0d6d43f5ac9c89cdb26120a7, 2ae1b689437747778b799d7b2263e75c]
id: PTKNfSUd01dU
outputId: 63cbdf61-8846-4465-aad2-b50751ae89cc
---
# Load CelebA dataset from HF
dataset = load_dataset("nielsr/CelebA-faces")
```

```{code-cell} ipython3
:id: eKccExxh037G


```

+++ {"id": "NNCY6qTE06P4"}

## Write TFRecords

* We save image as string (bytes) along with original size and dimension, which helps us when we read it back

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: KE_QD9MJ06on
outputId: d3f4c17e-2065-40d3-8f06-75a82f32b023
---
def parse_train():
    for item in dataset['train']:
        image = np.asarray(item['image'])
        height, width, channels = image.shape
        image_string = image.tobytes()
        yield {'image': image_string, 
               'height': height,
               'width': width,
               'channels': channels
              }
        
# Write using TF Writer

schema = {
    "image": ("var_len", "bytes"),
    "height": ("var_len", "int"),
    "width": ("var_len", "int"),
    "channels": ("var_len", "int")
}


tfrecord_train_dir = 'TFRECORDS/celeba'
tfrecord_filename = 'celeba'

tfwriter = TFWriter(schema=schema, 
                    file_name=tfrecord_filename, 
                    model_dir=tfrecord_train_dir,
                    tag='train',
                    overwrite=True,
                    verbose_counter=10000
                    )

# Train dataset
train_parser_fn = parse_train()
tfwriter.process(parse_fn=train_parser_fn)
```

```{code-cell} ipython3
:id: nlgmLbdZ1JZb


```

+++ {"id": "9O1LDkKE1RgA"}

## Read TFRecords

```{code-cell} ipython3
:id: LbGRxS_91Qm-

# Read TFRecord

schema = json.load(open("{}/schema.json".format(tfrecord_train_dir)))
all_files = glob.glob("{}/*.tfrecord".format(tfrecord_train_dir))
tf_reader = TFReader(schema=schema, 
                    tfrecord_files=all_files)

x_keys = ['image', 'height', 'width', 'channels']

def decode_img(item):
    byte_string = item['image'][0]
    im_height = item['height'][0]
    im_width = item['width'][0]
    im_channels = item['channels'][0]
    
    image_array = tf.io.decode_raw(byte_string, tf.uint8)
    image = tf.reshape(image_array, (im_height, im_width, im_channels))
    # image = tf.cast(tf.image.resize(image, (64, 64)), tf.uint8)
    return {'image': image}

batch_size = 16
train_dataset = tf_reader.read_record( 
                                   keys=x_keys,
                                   shuffle=True
                                  )

train_dataset = train_dataset.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.padded_batch(batch_size, drop_remainder=True)

for item in train_dataset:
    break
```

```{code-cell} ipython3
:id: Jenb2yrf1WAJ


```

+++ {"id": "3QG8w2rw1WyC"}

## Plot images after reading

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 699
id: i9E51VBm1Z8I
outputId: bf0b164c-fb67-4cca-e062-90b8bddb83fa
---
def display_images(images, cols=5):
    """Display given images and their labels in a grid."""
    rows = int(math.ceil(len(images) / cols))
    fig = plt.figure()
    fig.set_size_inches(cols * 3, rows * 3)
    for i, (image) in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(image)
        plt.title(i)

NUM_IMAGES = 16
# Extract each images individually for plot
batch_images = [im.numpy() for im in item['image']]
display_images(batch_images)
```

```{code-cell} ipython3
:id: OJKOISIW85fj


```
