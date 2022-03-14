#!/usr/bin/env python
# coding: utf-8

# # Classify text (MRPC) with Albert
# 
# This tutorial contains complete code to fine-tune Albert to perform binary classification on (MRPC) dataset. 
# In addition to training a model, you will learn how to preprocess text into an appropriate format.
# 
# In this notebook, you will:
# 
# - Load the MRPC dataset from HuggingFace
# - Load Albert Model using tf-transformers
# - Build train and validation dataset (on the fly) feature preparation using
# tokenizer from tf-transformers.
# - Build your own model by combining Albert with a classifier
# - Train your own model, fine-tuning Albert as part of that
# - Save your model and use it to classify sentences
# - Use the end-to-end (preprocessing + inference) in production setup
# 
# If you're new to working with the MNLI dataset, please see [MRPC](https://huggingface.co/datasets/glue/viewer/mrpc) for more details.

# In[ ]:


get_ipython().system('pip install tf-transformers')

get_ipython().system('pip install sentencepiece')

get_ipython().system('pip install tensorflow-text')

get_ipython().system('pip install transformers')

get_ipython().system('pip install wandb')

get_ipython().system('pip install datasets')


# In[ ]:





# In[3]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
import tensorflow_text as tf_text
import datasets
import wandb

print("Tensorflow version", tf.__version__)
print("Tensorflow text version", tf_text.__version__)
print("Devices", tf.config.list_physical_devices())

from tf_transformers.models import AlbertModel, Classification_Model, AlbertTokenizerTFText
from tf_transformers.core import Trainer
from tf_transformers.optimization import create_optimizer
from tf_transformers.losses.loss_wrapper import get_1d_classification_loss


# In[ ]:





# ### Load Model, Optimizer , Trainer
# 
# Our Trainer expects ```model```, ```optimizer``` and ```loss``` to be a function.

# In[4]:


# Load Model
def get_model(model_name, num_classes, is_training, use_dropout):
  """Get Model"""

  def model_fn():
    model = AlbertModel.from_pretrained(model_name)
    model = Classification_Model(model, num_classes=num_classes, is_training=is_training, use_dropout=use_dropout)
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


# In[ ]:





# ### Prepare Data for Training
# 
# We will make use of ```Tensorflow Text``` based tokenizer to do ```on-the-fly``` preprocessing, without having any
# overhead of pre prepapre the data in the form of ```pickle```, ```numpy``` or ```tfrecords```.

# In[5]:


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
        input_ids = tokenizer_layer({'text': item['sentence1'] + '  ' + item['sentence2']})
        # Truncate to max_seq_len-2 (2 is for CLS and SEP)
        input_ids = input_ids[:, :max_seq_len-2]
        # Add CLS and SEP
        input_ids = tf_text.combine_segments(
                      [input_ids], start_of_sequence_id=tokenizer_layer.cls_token_id, end_of_segment_id=tokenizer_layer.sep_token_id
                  )[0]
        input_ids, input_mask = tf_text.pad_model_inputs(input_ids, max_seq_length=max_seq_len)

        result = {}
        result['input_ids'] = input_ids
        result['input_mask'] = input_mask
        result['input_type_ids'] = tf.zeros_like(input_ids)

        labels = {}
        labels['labels'] = tf.expand_dims(item['label'], 1)
        return result, labels

    tfds_dict = dataset.to_dict()
    tfdataset = tf.data.Dataset.from_tensor_slices(tfds_dict).shuffle(100)

    tfdataset = tfdataset.batch(batch_size, drop_remainder=drop_remainder)
    tfdataset = tfdataset.map(parse, num_parallel_calls =tf.data.AUTOTUNE)
    
    # Shard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    tfdataset = tfdataset.with_options(options)
    
    return tfdataset


# In[ ]:





# ### Prepare Dataset
# 
# 1. Set necessay hyperparameters.
# 2. Prepare ```train dataset```, ```validation dataset```.
# 3. Load ```model```, ```optimizer```, ```loss``` and ```trainer```.

# In[6]:


# Data configs
dataset_name = 'mrpc'
model_name  = 'albert-base-v2'
max_seq_len = 128
batch_size  = 8

# Model configs
learning_rate = 1e-5
epochs = 4
num_classes = 2
model_checkpoint_dir = 'MODELS/mrpc_albert_model'

# Load HF dataset
dataset = datasets.load_dataset('glue', dataset_name)
# Load tokenizer from tf-transformers
tokenizer_layer = AlbertTokenizerTFText.from_pretrained(model_name)
# Train Dataset
train_dataset = load_dataset(dataset['train'], tokenizer_layer, max_seq_len, batch_size, drop_remainder=True)
# Validation Dataset
validation_dataset = load_dataset(dataset['validation'], tokenizer_layer, max_seq_len, batch_size, drop_remainder=False)

# Total train examples
total_train_examples = dataset['train'].num_rows
steps_per_epoch = total_train_examples // batch_size

# model
model_fn =  get_model(model_name, num_classes, is_training=True, use_dropout=True)
# optimizer
optimizer_fn = get_optimizer(learning_rate, total_train_examples, batch_size, epochs)
# trainer
trainer = get_trainer(distribution_strategy='mirrored', num_gpus=1)
# loss
loss_fn = get_1d_classification_loss()


# In[ ]:





# ### Wandb configuration

# In[ ]:


project = "TUTORIALS"
display_name = "mrpc_albert_base_v2"
wandb.init(project=project, name=display_name)


# In[ ]:





# ### Accuracy Callback

# In[9]:


import tqdm
from sklearn.metrics import accuracy_score

METRICS = [tf.keras.metrics.Accuracy(name="accuracy", dtype=None)]


class AccuracyCallback:
    def __init__(self, label_column: str, prediction_column: str) -> None:

        self.label_column = label_column
        self.prediction_column = prediction_column
        self.metrics = METRICS

    def __call__(self, trainer_kwargs):

        validation_dataset_distributed = iter(
            trainer_kwargs["validation_dataset_distributed"]
        )
        model = trainer_kwargs["model"]
        wandb = trainer_kwargs["wandb"]
        step = trainer_kwargs["global_step"]
        strategy = trainer_kwargs["strategy"]
        epoch = trainer_kwargs["epoch"]
        epochs = trainer_kwargs["epochs"]
        validation_steps = trainer_kwargs["validation_steps"]

        if validation_dataset_distributed is None:
            raise ValueError(
                "No validation dataset has been provided either in the trainer class, \
                                 or when callback is initialized. Please provide a validation dataset"
            )

        @tf.function
        def validate_run(dist_inputs):
            batch_inputs, batch_labels = dist_inputs
            model_outputs = model(batch_inputs)
            return tf.argmax(
                model_outputs[self.prediction_column], axis=1
            ), tf.reduce_max(model_outputs[self.prediction_column], axis=1)

        P_ids_flattened = []
        O_ids_flattened = []
        # This is a hack to make tqdm to print colour bar
        # TODO: fix it .
        pbar = tqdm.trange(validation_steps, colour="magenta", unit="batch")
        for step_counter in pbar:
            dist_inputs = next(validation_dataset_distributed)
            predicted_ids, predicted_probs = strategy.run(
                validate_run, args=(dist_inputs,)
            )
            predicted_ids = tf.concat(
                trainer.distribution_strategy.experimental_local_results(predicted_ids),
                axis=0,
            )
            predicted_probs = tf.concat(
                trainer.distribution_strategy.experimental_local_results(
                    predicted_probs
                ),
                axis=0,
            )

            # 1 in tuple of dist_inputs
            batch_labels = dist_inputs[1]
            original_ids = tf.squeeze(
                tf.concat(
                    trainer.distribution_strategy.experimental_local_results(
                        batch_labels[self.label_column]
                    ),
                    axis=0,
                ),
                axis=1,
            )
            P_ids_flattened.extend(predicted_ids)
            O_ids_flattened.extend(original_ids)
            metric_result = {}
            for metric in self.metrics:
                metric.update_state(original_ids, predicted_ids)
                metric_result[metric.name] = metric.result().numpy()
            pbar.set_description(
                "Callback: Epoch {}/{} --- Step {}/{} ".format(
                    epoch, epochs, step_counter, validation_steps
                )
            )
            pbar.set_postfix(**metric_result)
        # Result over whole dataset and reset
        metrics_result = {}
        for metric in self.metrics:
            metrics_result[metric.name] = metric.result().numpy()
            metric.reset_state()
        if wandb:
            wandb.log(metrics_result, step=step)

        metrics_result['acc_sklearn'] = accuracy_score(O_ids_flattened, P_ids_flattened)
        return metrics_result


# In[10]:





# ### Train :-)

# In[11]:


accuracy_callback = AccuracyCallback(label_column='labels', 
                                    prediction_column='class_logits')
history = trainer.run(
    model_fn=model_fn,
    optimizer_fn=optimizer_fn,
    train_dataset=train_dataset,
    train_loss_fn=loss_fn,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    model_checkpoint_dir=model_checkpoint_dir,
    batch_size=batch_size,
    validation_dataset=validation_dataset,
    validation_loss_fn=loss_fn,
    callbacks=[accuracy_callback],
    wandb=None
)


# ### Visualize Tensorboard

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')

get_ipython().run_line_magic('tensorboard', '--logdir MODELS/mrpc_albert_model/logs')


# In[ ]:





# ### Save and Serialize Model

# In[ ]:


# Save serialized version of the model

# Note: Ignore checkpoint warnings, it is because we save optimizer with checkpoint
# while we restoring, we take only model.

model = model_fn()
model.load_checkpoint(model_checkpoint_dir)

model.save_transformers_serialized('{}/saved_model/'.format(model_checkpoint_dir))


# In[13]:





# ### Model Serialization (Production)

# In[17]:


# Load serialized model

loaded = tf.saved_model.load("{}/saved_model/".format(model_checkpoint_dir))
model = loaded.signatures['serving_default']

# Lets evaluate accuracy and see whether it matches the callback

accuracy_metric = tf.keras.metrics.Accuracy('accuracy')
for (batch_inputs, batch_labels) in tqdm.tqdm(validation_dataset):
  model_outputs = model(**batch_inputs)
  predicted_ids = tf.argmax(model_outputs['class_logits'], axis=1)
  label_ids = batch_labels['labels']
  accuracy_metric.update_state(label_ids, predicted_ids) 
print("Validation Accuracy", accuracy_metric.result().numpy())


# ### Advanced Serialization (Include pre-processing with models)

# In[ ]:


# Advanced serialzation
from tf_transformers.core import ClassificationChainer
model = model_fn()
model.load_checkpoint(model_checkpoint_dir)

# Serialize tokenizer and model together
tokenizer_layer = AlbertTokenizerTFText.from_pretrained(
            model_name, add_special_tokens=True, max_length=max_seq_len, dynamic_padding=True, truncate=True
        )
model = ClassificationChainer(tokenizer_layer.get_model(), model)
model = model.get_model() # get_model will return tf.keras.Model , nothing fancy

model.save_serialized('{}/saved_model_text_model/'.format(model_checkpoint_dir)) # Do not use `model_transformers_serialzed` here


# In[ ]:





# ### Load the model + Check Accuracy
# 
# 1. The accuracy of the validation data matches, mean our pre-processing is right.
# 2. This also avoids pre-processing skew and make deployment easier.

# In[40]:


# Load jointly serialized model

loaded = tf.saved_model.load("{}/saved_model_text_model/".format(model_checkpoint_dir))
model = loaded.signatures['serving_default']

# Now lets evaluate accuracy again
# This time, we have to provide only raw text, model will be tokenizing it internally

# Create a validation dataset
validation_text_dataset = []
validation_labels = []
for item in dataset['validation']:
  validation_text_dataset.append(item['sentence1'] + ' ' + item['sentence2'])
  validation_labels.append(item['label'])

validation_text_dataset = tf.data.Dataset.from_tensor_slices(({'text': validation_text_dataset}, 
                                                             {'labels': validation_labels})
                                                             ).batch(batch_size)

# Evaluate accuracy
accuracy_metric = tf.keras.metrics.Accuracy('accuracy')
for (batch_inputs, batch_labels) in tqdm.tqdm(validation_text_dataset):
  model_outputs = model(**batch_inputs)
  predicted_ids = tf.argmax(model_outputs['class_logits'], axis=1)
  label_ids = batch_labels['labels']
  accuracy_metric.update_state(label_ids, predicted_ids) 
print("Validation Accuracy", accuracy_metric.result().numpy())


# In[ ]:




