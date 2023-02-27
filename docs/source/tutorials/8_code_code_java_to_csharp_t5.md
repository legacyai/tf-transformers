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

# Code Java to C# using T5

This tutorial contains complete code to fine-tune T5 to perform Seq2Seq on CodexGLUE Code to Code dataset. 
In addition to training a model, you will learn how to preprocess text into an appropriate format.

In this notebook, you will:

- Load the CodexGLUE code to code dataset from HuggingFace
- Load T5 Model using tf-transformers
- Build train and validation dataset (on the fly) feature preparation using
tokenizer from tf-transformers.
- Train your own model, fine-tuning T5 as part of that
- Evaluate BLEU on the generated text 
- Save your model and use it to convert Java to C# sentences
- Use the end-to-end (preprocessing + inference) in production setup

If you're new to working with the CodexGLUE dataset, please see [CodexGLUE](https://huggingface.co/datasets/code_x_glue_cc_code_to_code_trans) for more details.

```{code-cell} ipython3

```

```{code-cell} ipython3
!pip install tf-transformers

!pip install sentencepiece

!pip install tensorflow-text

!pip install transformers

!pip install wandb

!pip install datasets
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
import tensorflow_text as tf_text
import datasets
import tqdm
import wandb

print("Tensorflow version", tf.__version__)
print("Tensorflow text version", tf_text.__version__)
print("Devices", tf.config.list_physical_devices())

from tf_transformers.models import T5Model, T5TokenizerTFText
from tf_transformers.core import Trainer
from tf_transformers.optimization import create_optimizer
from tf_transformers.losses import cross_entropy_loss_label_smoothing
from tf_transformers.text import TextDecoder
```

```{code-cell} ipython3

```

### Load Model, Optimizer , Trainer

Our Trainer expects ```model```, ```optimizer``` and ```loss``` to be a function.

```{code-cell} ipython3
# Load Model
def get_model(model_name, is_training, use_dropout):
  """Get Model"""

  def model_fn():
    model = T5Model.from_pretrained(model_name)
    return model
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

# Load loss
def loss_fn(y_true_dict, y_pred_dict, smoothing=0.1):
    
    loss = cross_entropy_loss_label_smoothing(labels=y_true_dict['labels'], 
                                   logits=y_pred_dict['token_logits'],
                                   smoothing=smoothing,
                                      label_weights=y_true_dict['labels_mask'])
    return {'loss': loss}
```

```{code-cell} ipython3

```

### Prepare Data for Training

We will make use of ```Tensorflow Text``` based tokenizer to do ```on-the-fly``` preprocessing, without having any
overhead of pre prepapre the data in the form of ```pickle```, ```numpy``` or ```tfrecords```.

```{code-cell} ipython3
# Load dataset
def load_dataset(dataset, tokenizer_layer, max_seq_len, batch_size, drop_remainder):
    """
    Args:
      dataset; HuggingFace dataset
      tokenizer_layer: tf-transformers tokenizer
      max_seq_len: int (maximum sequence length of text)
      batch_size: int (batch_size)
      drop_remainder: bool (to drop remaining batch_size, when its uneven)
    """
    def parse(item):
        # Encoder inputs
        encoder_input_ids = tokenizer_layer({'text': [item['java']]})
        encoder_input_ids = encoder_input_ids.merge_dims(-2, 1)
        encoder_input_ids = encoder_input_ids[:max_seq_len-1]
        encoder_input_ids = tf.concat([encoder_input_ids, [tokenizer_layer.eos_token_id]], axis=0)

        # Decoder inputs
        decoder_input_ids = tokenizer_layer({'text': [item['cs']]})
        decoder_input_ids = decoder_input_ids.merge_dims(-2, 1)
        decoder_input_ids = decoder_input_ids[:max_seq_len-2]
        decoder_input_ids = tf.concat([[tokenizer_layer.pad_token_id] , decoder_input_ids, [tokenizer_layer.eos_token_id]], axis=0)


        encoder_input_mask = tf.ones_like(encoder_input_ids)
        labels = decoder_input_ids[1:]
        labels_mask = tf.ones_like(labels)
        decoder_input_ids = decoder_input_ids[:-1]

        result = {}
        result['encoder_input_ids'] = encoder_input_ids
        result['encoder_input_mask'] = encoder_input_mask
        result['decoder_input_ids'] = decoder_input_ids

        labels_dict = {}
        labels_dict['labels'] = labels
        labels_dict['labels_mask'] = labels_mask
        return result, labels_dict

    tfds_dict = dataset.to_dict()
    tfdataset = tf.data.Dataset.from_tensor_slices(tfds_dict).shuffle(100)

    tfdataset = tfdataset.map(parse, num_parallel_calls =tf.data.AUTOTUNE)
    tfdataset = tfdataset.padded_batch(batch_size, drop_remainder=drop_remainder)

    # Shard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    tfdataset = tfdataset.with_options(options)
    
    return tfdataset
```

```{code-cell} ipython3

```

### Prepare Dataset

1. Set necessay hyperparameters.
2. Prepare ```train dataset```, ```validation dataset```.
3. Load ```model```, ```optimizer```, ```loss``` and ```trainer```.

```{code-cell} ipython3

```

```{code-cell} ipython3
# Data configs
dataset_name = 'code_x_glue_cc_code_to_code_trans'
model_name  = 't5-small'
max_seq_len = 256
batch_size  = 32

# Model configs
learning_rate = 1e-4
epochs = 10
model_checkpoint_dir = 'MODELS/t5_code_to_code'

# Load HF dataset
dataset = datasets.load_dataset(dataset_name)
# Load tokenizer from tf-transformers
tokenizer_layer = T5TokenizerTFText.from_pretrained(model_name)
# Train Dataset
train_dataset = load_dataset(dataset['train'], tokenizer_layer, max_seq_len, batch_size, drop_remainder=True)
# Validation Dataset
validation_dataset = load_dataset(dataset['test'], tokenizer_layer, max_seq_len, batch_size, drop_remainder=False)

# Total train examples
total_train_examples = dataset['train'].num_rows
steps_per_epoch = total_train_examples // batch_size

# model
model_fn =  get_model(model_name, is_training=True, use_dropout=True)
# optimizer
optimizer_fn = get_optimizer(learning_rate, total_train_examples, batch_size, epochs, use_constant_lr=True)
# trainer
trainer = get_trainer(distribution_strategy='mirrored', num_gpus=1)
```

```{code-cell} ipython3

```

### Wandb Configuration

```{code-cell} ipython3
project = "TUTORIALS"
display_name = "t5_code_to_code"
wandb.init(project=project, name=display_name)
```

```{code-cell} ipython3

```

### Train :-)

```{code-cell} ipython3
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

```{code-cell} ipython3

```

### Load and Serialize Model for Text Generation
* 1. Load T5Model with ```use_auto_regressive=True```

```{code-cell} ipython3
# Load T5 for Auto Regressive
model = T5Model.from_pretrained(model_name, use_auto_regressive=True)
# Load from checkpoint dir
model.load_checkpoint(model_checkpoint_dir)
# Save and serialize
model.save_transformers_serialized('{}/saved_model'.format(model_checkpoint_dir), overwrite=True)
# Load model
loaded = tf.saved_model.load('{}/saved_model'.format(model_checkpoint_dir))
```

### Evaluate on Test ( BLEU ) score

```{code-cell} ipython3
# Load decoder
decoder = TextDecoder(model=loaded)

# greedy decoding
predicted_text = []
original_text  = []
for (batch_inputs, batch_labels) in tqdm.tqdm(validation_dataset):
    
    decoder_input_ids = batch_inputs['decoder_input_ids']
    
    # While decoding we do not need this, decoder_start_token_id will be automatically taken from saved model
    del batch_inputs['decoder_input_ids']
    
    predictions = decoder.decode(batch_inputs, 
                             mode='greedy', 
                             max_iterations=max_seq_len, 
                             eos_id=tokenizer_layer.eos_token_id)
    # Decode predictions
    predicted_text_batch = tokenizer_layer._tokenizer.detokenize(tf.cast(tf.squeeze(predictions['predicted_ids'], axis=1), tf.int32))
    predicted_text_batch = [text.numpy().decode() for text in predicted_text_batch]
    predicted_text.extend(predicted_text_batch)
    
    # Decoder original text
    original_text_batch = tokenizer_layer._tokenizer.detokenize(decoder_input_ids)
    original_text_batch = [text.numpy().decode() for text in original_text_batch]
    original_text.extend(original_text_batch)
    
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import sacrebleu
from sacremoses import MosesDetokenizer
# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(predicted_text, [original_text])
print("BLEU: {}".format(bleu.score))
```

```{code-cell} ipython3

```
