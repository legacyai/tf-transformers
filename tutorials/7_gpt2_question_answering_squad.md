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

# GPT2 for QA using Squad V1 ( Causal LM )

This tutorial contains complete code to fine-tune GPT2 to finetune for Question Answering using Squad V1 data.
In addition to training a model, you will learn how to preprocess text into an appropriate format.

In this notebook, you will:

- Load the Squad v1 dataset from HuggingFace
- Load GPT2 Model using tf-transformers
- Build model using ```causal``` (default) and ```prefix``` masking.
- Build train and validation dataset  feature preparation using
tokenizer from transformers.
- Train your own model, fine-tuning GPT2 
- Save your model and use it to for QA
- Use the end-to-end (inference) in production setup

If you're new to working with the Quora dataset, please see [SQUAD](https://huggingface.co/datasets/squad) for more details.

```{code-cell} ipython3
!pip install tf-transformers

!pip install transformers

!pip install wandb

!pip install datasets
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import tensorflow as tf
import random
import collections
import wandb
import tempfile
import tqdm
import json

import os
import numpy as np

print("Tensorflow version", tf.__version__)
print("Devices", tf.config.list_physical_devices())

from tf_transformers.models import GPT2Model
from tf_transformers.core import Trainer
from tf_transformers.optimization import create_optimizer
from tf_transformers.data import TFWriter, TFReader
from tf_transformers.losses.loss_wrapper import get_lm_loss
from tf_transformers.text import TextDecoder


from datasets import load_dataset


from transformers import GPT2Tokenizer
```

```{code-cell} ipython3

```

### Load Data, Tokenizer

```{code-cell} ipython3
model_name = 'gpt2'

# Load Dataset
dataset = load_dataset("squad")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define length for examples
max_sequence_length = 384
max_question_length = 64
max_answer_length = 40
batch_size = 32
```

### Prepare Training TFRecords and Validation TFRecords using Squad ( causal and prefix )

* 1. We combine ```(question + context + answer)```
* 2. For ```mask_mode=causal```, we don't need any mask. For ```mask_mode=prefix```, we need ```input_mask```.
* 3. For ```prefix```, we will mask only ```question + context```, as ```answer``` is supposed to be generated, we shouldn't mask it, means its causal.
* 4. Note how ```labels_mask``` is prepared and how it is different from ```input_mask```.

```{code-cell} ipython3
def parse_train(dataset, tokenizer, max_question_length, max_passage_length, max_answer_length, key):
    """Function to parse examples which are is_duplicate=1

    Args:
        dataset (:obj:`dataet`): HF dataset
        tokenizer (:obj:`tokenizer`): HF Tokenizer
        max_question_length (:obj:`int`): Question Length
        max_passage_length (:obj:`int`): Passage Length
        max_answer_length (:obj:`int`): Answer Length
        key (:obj:`str`): Key of dataset (`train`, `validation` etc)
    """    
    result = {}
    for f in dataset[key]:
        
        question_ids = tokenizer('Question: ' + f['question'], max_length=max_question_length, truncation=True)['input_ids']
        context_ids  = tokenizer('Context: ' + f['context'], max_length=max_passage_length, truncation=True)['input_ids']
        answer_ids   = tokenizer('answer: ' + f['answers']['text'][0], max_length=max_answer_length, truncation=True)['input_ids']
        # add EOS
        context_ids = context_ids + [tokenizer.bos_token_id]
        answer_ids  = answer_ids + [tokenizer.bos_token_id] # EOS token
        
        # input_ids
        input_ids = (question_ids + context_ids + answer_ids)
        
        # input_mask
        input_mask = ([1] * len(question_ids)) + ([1] * len(context_ids)) + ([0] * len(answer_ids))
        # labels mask is opposite to input_mask, as we need to find loss only on answerids
        labels_mask = ([0] * len(question_ids)) + ([0] * len(context_ids)) + ([1] * len(answer_ids))
        result = {}
        # Except last word
        result['input_ids'] = input_ids[:-1]
        result['input_mask'] = input_mask[:-1]
        
        # Shift one word next
        result['labels'] = input_ids[1:]
        result['labels_mask'] = labels_mask[1:]
        
        yield result
        
# Write using TF Writer
schema = {
    "input_ids": ("var_len", "int"),
    "input_mask": ("var_len", "int"),
    "labels": ("var_len", "int"),
    "labels_mask": ("var_len", "int")
    
}

tfrecord_train_dir = tempfile.mkdtemp()
tfrecord_filename = 'squad'

tfwriter = TFWriter(schema=schema, 
                    file_name=tfrecord_filename, 
                    model_dir=tfrecord_train_dir,
                    tag='train',
                    overwrite=True
                    )

# Train dataset
train_parser_fn = parse_train(dataset, tokenizer, max_question_length, max_sequence_length, max_answer_length, key='train')
tfwriter.process(parse_fn=train_parser_fn)
```

```{code-cell} ipython3

```

### Prepare Validation TFRecords

```{code-cell} ipython3
def parse_dev(dataset, tokenizer, max_question_length, max_passage_length, max_answer_length, key):
    """Function to parse examples
    Args:
        dataset (:obj:`dataet`): HF dataset
        tokenizer (:obj:`tokenizer`): HF Tokenizer
        max_question_length (:obj:`int`): Question Length
        max_passage_length (:obj:`int`): Passage Length
        max_answer_length (:obj:`int`): Answer Length
        key (:obj:`str`): Key of dataset (`train`, `validation` etc)
    """    
    result = {}
    for f in dataset[key]:
        
        question_ids = tokenizer('Question: ' + f['question'], max_length=max_question_length, truncation=True)['input_ids']
        context_ids  = tokenizer('Context: ' + f['context'], max_length=max_passage_length, truncation=True)['input_ids']
        answer_ids   = tokenizer('answer: ' + f['answers']['text'][0], max_length=max_answer_length, truncation=True)['input_ids']
        # add EOS
        context_ids = context_ids + [tokenizer.bos_token_id]
        
        # input_ids
        input_ids = (question_ids + context_ids)
        
        # input_mask
        input_mask = ([1] * len(question_ids)) + ([1] * len(context_ids))
        result['input_ids'] = input_ids
        result['input_mask'] = input_mask
        result['original_answer'] = f['answers']['text'][0]
        
        yield result
        
tfrecord_validation_dir = tempfile.mkdtemp()
tfrecord_validation_filename = 'squad'

validation_schema = {
    "input_ids": ("var_len", "int"),
    "input_mask": ("var_len", "int"),
    "original_answer": ("var_len", "bytes")
    
}
tfwriter = TFWriter(schema=validation_schema, 
                    file_name=tfrecord_validation_filename, 
                    model_dir=tfrecord_validation_dir,
                    tag='eval',
                    overwrite=True
                    )

# Validation dataset
validation_parser_fn = parse_dev(dataset, tokenizer, max_question_length, max_sequence_length, max_answer_length, key='validation')
tfwriter.process(parse_fn=validation_parser_fn)
```

### Wandb Configuration

```{code-cell} ipython3
project = "TUTORIALS"
display_name = 'causal_mask'
wandb.init(project=project, name=display_name)
```

### Load Model, Optimizer , Trainer

Our Trainer expects ```model```, ```optimizer``` and ```loss``` to be a function.

```{code-cell} ipython3
# Load Model
def get_model(model_name, is_training, use_dropout, mask_mode='causal'):
  """Get Model"""

  def model_fn():
    model = GPT2Model.from_pretrained(model_name, mask_mode=mask_mode) #causal by default
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

# Load loss fn
def get_loss():
    loss_fn = get_lm_loss(label_column='labels', 
                          label_weights_column='labels_mask')
    return loss_fn
```

```{code-cell} ipython3

```

### Set Hyperparameters and Configs

1. Set necessay hyperparameters.
2. Prepare ```train dataset```, ```validation dataset```.
3. Load ```model```, ```optimizer```, ```loss``` and ```trainer```.

```{code-cell} ipython3
# Model configs
learning_rate = 2e-5
epochs = 5
model_checkpoint_dir = 'MODELS/gpt2_squad_causal'

# Train dataset
schema = json.load(open("{}/schema.json".format(tfrecord_train_dir)))
total_train_examples = json.load(open("{}/stats.json".format(tfrecord_train_dir)))['total_records']


all_files = tf.io.gfile.glob("{}/*.tfrecord".format(tfrecord_train_dir))
tf_reader = TFReader(schema=schema, 
                    tfrecord_files=all_files)

x_keys = ['input_ids']
y_keys = ['labels', 'labels_mask']
train_dataset = tf_reader.read_record(auto_batch=True, 
                                   batch_size=batch_size, 
                                   x_keys = x_keys, 
                                   y_keys = y_keys,
                                   shuffle=True
                                  )

# Total train examples
steps_per_epoch = total_train_examples // batch_size

# model
model_fn =  get_model(model_name, is_training=True, use_dropout=True, mask_mode='causal')
# optimizer
optimizer_fn = get_optimizer(learning_rate, total_train_examples, batch_size, epochs)
# loss
loss_fn = get_loss()
# trainer (multi gpu strategy)
trainer = get_trainer(distribution_strategy='mirrored', num_gpus=2)
```

```{code-cell} ipython3

```

### Train GPT2 Causal :-)

* 1. Loss is coming down in epoch 1 itself.
* 2. Evaluation results clearly indicated how well model has learned.

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
    steps_per_call=1,
    wandb=wandb
)
```

### Evaluation Script (Squad V1) - Exact match, F1 score

```{code-cell} ipython3
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    """Squad evaluate
    
    dataset: Huggingface Dataset
    predictions: List of predictions
    """
    f1 = exact_match = total = 0
    for item in dataset:
        ground_truths = item['answers']['text'] # list of answers
        prediction = predictions[total]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}
```

```{code-cell} ipython3

```

### Evaluate ( exact match and F1 score ) on all checkpoints - GPT2 Causal

```{code-cell} ipython3
def split_by_id(predicted_ids, eos_id):
    """Split by EOS_ID to make decoding proper"""
    all_ids = []
    for per_example_id in predicted_ids:
        index = -1
        if eos_id in per_example_id:
            index = per_example_id.index(eos_id)
        sliced_ids = per_example_id[:index]
        all_ids.append(sliced_ids)
    return all_ids

def get_serialized_model_from_checkpoint(model_checkpoint_dir, checkpoint_number):
    """Load serialized model checkpoint"""
    model = GPT2Model.from_pretrained(model_name, use_auto_regressive=True)
    model.load_checkpoint(checkpoint_path='{}/ckpt-{}'.format(model_checkpoint_dir, checkpoint_number))

    model.save_transformers_serialized('{}/saved_model'.format(model_checkpoint_dir), overwrite=True)
    
    loaded = tf.saved_model.load('{}/saved_model'.format(model_checkpoint_dir))
    
    return loaded

# Validation dataset
validation_files = tf.io.gfile.glob("{}/*.tfrecord".format(tfrecord_validation_dir))
tf_reader = TFReader(schema=validation_schema, 
                    tfrecord_files=validation_files)

x_keys = ['input_ids']
y_keys = ['original_answer'] # not necessarily required
validation_dataset = tf_reader.read_record(auto_batch=True, 
                                   batch_size=batch_size, 
                                   x_keys = x_keys, 
                                   y_keys = y_keys,
                                   shuffle=False,
                                   padded_values={'input_ids': tf.constant(-1)}
                                  )

validation_results = []
for checkpoint_number in range(1, epochs+1):
    
    # get serialized model
    loaded  = get_serialized_model_from_checkpoint(model_checkpoint_dir, checkpoint_number)
    
    # Load decoder
    decoder = TextDecoder(model=loaded)
    
    # greedy decoding
    predicted_answers = []
    for (batch_inputs, batch_labels) in tqdm.tqdm(validation_dataset):
        predictions = decoder.decode(batch_inputs, 
                                 mode='greedy', 
                                 max_iterations=max_answer_length, 
                                 eos_id=tokenizer.bos_token_id)

        predicted_ids = tf.squeeze(predictions['predicted_ids'], axis=1).numpy().tolist()
        # Squeeze to 2D
        predicted_ids = split_by_id(predicted_ids, tokenizer.bos_token_id)
        # Decode
        predicted_answers_batch = tokenizer.batch_decode(predicted_ids)
        predicted_answers.extend(predicted_answers_batch)
        
    # generation will start with 'answer:'. remove that
    predicted_answers = [answer.replace('answer: ', '') for answer in predicted_answers]
    # Exact match and f1 score
    val_result = evaluate(dataset['validation'], predicted_answers)
    validation_results.append(val_result)
    
for checkpoint_number, result in enumerate(validation_results):
    print("Checkpoint {} , {}".format(checkpoint_number+1, result))
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
