---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3
  name: python3
---

```{code-cell}
:id: jbxNfeNfhrIt


```

+++ {"id": "3oU2Xas2ikcw"}

# Train (Masked Language Model) with tf-transformers in TPU

This tutorial contains complete code to train MLM model on C4 EN 10K dataset.
In addition to training a model, you will learn how to preprocess text into an appropriate format.

In this notebook, you will:

- Load the C4 (10k EN) dataset from HuggingFace
- Load GPT2 style (configuration) Model using tf-transformers
- Build train dataset (on the fly) feature preparation using
tokenizer from tf-transformers.
- Build a masked LM Model from GPT2 style configuration
- Save your model
- Use the base model for further tasks

If you're new to working with the C4 dataset, please see [C4](https://www.tensorflow.org/datasets/catalog/c4) for more details.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 5V5sXDHbi4Fr
outputId: 8a2dcfd7-1b69-4182-a24f-f533fb9c41b5
---
!pip install tf-transformers

!pip install sentencepiece

!pip install tensorflow-text

!pip install transformers

!pip install wandb

!pip install datasets
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Ub8L8158jHhI
outputId: cc274b0e-9867-4609-b8a8-bc8b6e68d4e9
---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
import tensorflow_text as tf_text
import datasets
import wandb

print("Tensorflow version", tf.__version__)
print("Tensorflow text version", tf_text.__version__)
print("Devices", tf.config.list_physical_devices())

from tf_transformers.models import GPT2Model, MaskedLMModel, AlbertTokenizerTFText
from tf_transformers.core import Trainer
from tf_transformers.optimization import create_optimizer
from tf_transformers.text.lm_tasks import mlm_fn
from tf_transformers.losses.loss_wrapper import get_lm_loss
```

### Trainer has to be initialized before everything (sometimes).

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: yNgQDdsgj4Zw
outputId: d8d95b9e-047f-46e9-bd63-2f5a02495efc
---
trainer = Trainer(distribution_strategy='tpu', num_gpus=0, tpu_address='colab')
```

+++ {"id": "7MRKsHoyj_wU"}

### Load Model, Optimizer , Trainer

Our Trainer expects ```model```, ```optimizer``` and ```loss``` to be a function.

```{code-cell}
:id: JJ9KB3oCkAVP

# Load Model
def get_model(model_name, vocab_size, is_training, use_dropout, num_hidden_layers):
  """Get Model"""

  def model_fn():
    config = GPT2Model.get_config(model_name)
    config['vocab_size'] = vocab_size
    model = GPT2Model.from_config(config, mask_mode='user_defined', num_hidden_layers=num_hidden_layers, return_layer=True)
    model = MaskedLMModel(
              model,
              use_extra_mlm_layer=False,
              hidden_size=config['embedding_size'],
              layer_norm_epsilon=config['layer_norm_epsilon'],
          )    
    return model.get_model()
  return model_fn

# Load Optimizer
def get_optimizer(learning_rate, examples, batch_size, epochs, use_constant_lr=False):
    """Get optimizer"""
    steps_per_epoch = int(examples / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * num_train_steps)

    def optimizer_fn():
        optimizer, learning_rate_fn = create_optimizer(learning_rate, num_train_steps, warmup_steps, use_constant_lr=use_constant_lr)
        return optimizer

    return optimizer_fn

# Load trainer
def get_trainer(distribution_strategy, num_gpus=0, tpu_address=None):
    """Get Trainer"""
    trainer = Trainer(distribution_strategy, num_gpus=num_gpus, tpu_address=tpu_address)
    return trainer
```

+++ {"id": "FrbiylDkklJ1"}

### Prepare Data for Training

We will make use of ```Tensorflow Text``` based tokenizer to do ```on-the-fly``` preprocessing, without having any
overhead of pre prepapre the data in the form of ```pickle```, ```numpy``` or ```tfrecords```.

```{code-cell}
:id: DPfht1IOklYA

# Load dataset
def load_dataset(dataset, tokenizer_layer, max_seq_len, max_predictions_per_seq, batch_size):
    """
    Args:
      dataset; HuggingFace dataset
      tokenizer_layer: tf-transformers tokenizer
      max_seq_len: int (maximum sequence length of text)
      batch_size: int (batch_size)
      max_predictions_per_seq: int (Maximum number of words to mask)
    """
    tfds_dict = dataset.to_dict()
    tfdataset = tf.data.Dataset.from_tensor_slices(tfds_dict)

    # MLM function
    masked_lm_map_fn = mlm_fn(tokenizer_layer, max_seq_len, max_predictions_per_seq)

    # MLM
    tfdataset = tfdataset.map(masked_lm_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # Batch
    tfdataset = tfdataset.batch(batch_size, drop_remainder=True).shuffle(50)

    # Auto SHARD
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    tfdataset = tfdataset.with_options(options)
    
    return tfdataset
```

+++ {"id": "Q_dFlI_MrDwG"}

### Prepare Dataset

1. Set necessay hyperparameters.
2. Prepare ```train dataset```
3. Load ```model```, ```optimizer```, ```loss``` and ```trainer```.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 104
  referenced_widgets: [d5c1adfe09f7461c89e172b9743af081, f729259d2bbb445888be1143cc4ffe4a,
    c57a7dce76fe4235a9961df2b256ee62, 90b4ff5b71554b5fa21f304e18407da1, 55d6f70eef0c484ca3014e74e7520eb6,
    9e7d5627b0ed4d2a953cc7a64f169817, 56d29164ccb643bf9879813c90281151, 7f567c87ab0d46b782796f6bdec74689,
    9ba4bc085776410889c4bfb5cef623e5, a7aebc9e539049ba9c86728a5d402cf2, fdd587ac1984477485a714a9be9bb2a9]
id: s_HE4DquolW2
outputId: 3f22ea55-1572-4670-bd69-8702153f99ff
---
# Data configs
dataset_name = 'stas/c4-en-10k'
model_name  = 'gpt2'
max_seq_len = 128
max_predictions_per_seq = 20
batch_size  = 128

# Model configs
learning_rate = 0.0005
epochs = 3
model_checkpoint_dir = 'gs://legacyai-bucket/sample_mlm_model' # If using TPU, provide GCP bucket for 
                                                        # storing model checkpoints

# Load HF dataset
dataset = datasets.load_dataset(dataset_name)
# Load tokenizer from tf-transformers
tokenizer_layer = AlbertTokenizerTFText.from_pretrained("albert-base-v2")
# Train Dataset
train_dataset = load_dataset(dataset['train'], tokenizer_layer, max_seq_len, max_predictions_per_seq, batch_size)

# Total train examples
total_train_examples = dataset['train'].num_rows
steps_per_epoch = 5000
num_hidden_layers = 8

# model
vocab_size = tokenizer_layer.vocab_size.numpy()
model_fn =  get_model(model_name, vocab_size, is_training=True, use_dropout=True, num_hidden_layers=num_hidden_layers)
# optimizer
optimizer_fn = get_optimizer(learning_rate, total_train_examples, batch_size, epochs, use_constant_lr=True)
# trainer
# trainer = get_trainer(distribution_strategy='tpu', num_gpus=0, tpu_address='colab')
# loss
loss_fn = get_lm_loss(loss_type=None)
```

+++ {"id": "6QUkKfQoscLF"}

### Wandb configuration

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 125
id: g9EZbosAqbIG
outputId: 699394c7-c9d9-457f-f48b-77627988b22c
---
project = "TUTORIALS"
display_name = "mlm_tpu"
wandb.init(project=project, name=display_name)
```

```{code-cell}
:id: LKZugOHYUmC5


```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Gexhy6NplTv7
outputId: 6010d4c6-68ec-4ac2-af8c-e5cebf14bd3e
---
history = trainer.run(
    model_fn=model_fn,
    optimizer_fn=optimizer_fn,
    train_dataset=train_dataset,
    train_loss_fn=loss_fn,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    model_checkpoint_dir=model_checkpoint_dir,
    batch_size=batch_size,
    wandb=wandb
)
```

+++ {"id": "ZuZefg0g2M50"}

### Load the Model from checkpoint

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: A0pi3h222NBm
outputId: 539a28d6-0e7e-42b6-896c-d1301cb7ccd6
---
model_fn =  get_model(model_name, vocab_size, is_training=False, use_dropout=False, num_hidden_layers=num_hidden_layers)

model = model_fn()
model.load_checkpoint(model_checkpoint_dir)
```

```{code-cell}
:id: H8KYSFqqQcHV


```

+++ {"id": "_oDAmWjlQcK0"}

### Test Model performance 

1. We can assess model performance by checking how it predicts masked word on sample sentences.
2. As we see the following result, its clear that model starts learning.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: bRoOpRV93Bun
outputId: 52fa7b71-96a8-489a-815f-3e65a20dd3a9
---
from transformers import AlbertTokenizer
tokenizer_hf = AlbertTokenizer.from_pretrained("albert-base-v2")

validation_sentences = [
    'Read the rest of this [MASK] to understand things in more detail.',
    'I want to buy the [MASK] because it is so cheap.',
    'The [MASK] was amazing.',
    'Sachin Tendulkar is one of the [MASK] palyers in the world.',
    '[MASK] is the capital of France.',
    'Machine Learning requires [MASK]',
    'He is working as a [MASK]',
    'She is working as a [MASK]',
]
inputs = tokenizer_hf(validation_sentences, padding=True, return_tensors="tf")

inputs_tf = {}
inputs_tf["input_ids"] = inputs["input_ids"]
inputs_tf["input_mask"] = inputs["attention_mask"]
seq_length = tf.shape(inputs_tf['input_ids'])[1]
inputs_tf['masked_lm_positions'] = tf.zeros_like(inputs_tf["input_ids"]) + tf.range(seq_length)


top_k = 10 # topk similar words
outputs_tf = model(inputs_tf)
# Get masked positions from each sentence
masked_positions = tf.argmax(tf.equal(inputs_tf["input_ids"], tokenizer_hf.mask_token_id), axis=1)
for i, logits in enumerate(outputs_tf['token_logits']):
    mask_token_logits = logits[masked_positions[i]]
    # 0 for probs and 1 for indexes from tf.nn.top_k
    top_words = tokenizer_hf.decode(tf.nn.top_k(mask_token_logits, k=top_k)[1].numpy())
    print("Input ----> {}".format(validation_sentences[i]))
    print("Predicted words ----> {}".format(top_words.split()))
    print()
```

```{code-cell}
:id: 80Y0ipAP4K5k


```
