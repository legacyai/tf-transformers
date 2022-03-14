#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Train (Masked Language Model) with tf-transformers in TPU
# 
# This tutorial contains complete code to train MLM model on C4 EN 10K dataset.
# In addition to training a model, you will learn how to preprocess text into an appropriate format.
# 
# In this notebook, you will:
# 
# - Load the C4 (10k EN) dataset from HuggingFace
# - Load GPT2 style (configuration) Model using tf-transformers
# - Build train dataset (on the fly) feature preparation using
# tokenizer from tf-transformers.
# - Build a masked LM Model from GPT2 style configuration
# - Save your model
# - Use the base model for further tasks
# 
# If you're new to working with the C4 dataset, please see [C4](https://www.tensorflow.org/datasets/catalog/c4) for more details.

# In[ ]:


get_ipython().system('pip install tf-transformers')

get_ipython().system('pip install sentencepiece')

get_ipython().system('pip install tensorflow-text')

get_ipython().system('pip install transformers')

get_ipython().system('pip install wandb')

get_ipython().system('pip install datasets')


# In[28]:


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


# ### Trainer has to be initialized before everything only in TPU (sometimes).

# In[30]:


trainer = Trainer(distribution_strategy='tpu', num_gpus=0, tpu_address='colab')


# ### Load Model, Optimizer , Trainer
# 
# Our Trainer expects ```model```, ```optimizer``` and ```loss``` to be a function.

# In[31]:


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


# ### Prepare Data for Training
# 
# We will make use of ```Tensorflow Text``` based tokenizer to do ```on-the-fly``` preprocessing, without having any
# overhead of pre prepapre the data in the form of ```pickle```, ```numpy``` or ```tfrecords```.

# In[32]:


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


# ### Prepare Dataset
# 
# 1. Set necessay hyperparameters.
# 2. Prepare ```train dataset```
# 3. Load ```model```, ```optimizer```, ```loss``` and ```trainer```.

# In[54]:


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


# ### Wandb configuration

# In[10]:


project = "TUTORIALS"
display_name = "mlm_tpu"
wandb.init(project=project, name=display_name)


# In[51]:





# ### Train :-)

# In[55]:


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


# ### Load the Model from checkpoint

# In[56]:


model_fn =  get_model(model_name, vocab_size, is_training=False, use_dropout=False, num_hidden_layers=num_hidden_layers)

model = model_fn()
model.load_checkpoint(model_checkpoint_dir)


# In[ ]:





# ### Test Model performance 
# 
# 1. We can assess model performance by checking how it predicts masked word on sample sentences.
# 2. As we see the following result, its clear that model starts learning.

# In[60]:


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


# In[ ]:




