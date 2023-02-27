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

# Create Sentence Embedding Roberta Model + Zeroshot from Scratch

This tutorial contains complete code to fine-tune Roberta to build meaningful sentence transformers using Quora Dataset from HuggingFace. 
In addition to training a model, you will learn how to preprocess text into an appropriate format.

In this notebook, you will:

- Load the Quora dataset from HuggingFace
- Load Roberta Model using tf-transformers
- Build train and validation dataset  feature preparation using
tokenizer from transformers.
- Build your own model by combining Roberta with a CustomWrapper
- Train your own model, fine-tuning Roberta as part of that
- Save your model and use it to extract sentence embeddings
- Use the end-to-end (inference) in production setup

If you're new to working with the Quora dataset, please see [QUORA](https://huggingface.co/datasets/quora) for more details.

```{code-cell} ipython3

```

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

from tf_transformers.models import RobertaModel, Classification_Model
from tf_transformers.core import Trainer
from tf_transformers.optimization import create_optimizer
from tf_transformers.data import TFWriter, TFReader
from tf_transformers.losses import cross_entropy_loss_for_classification

from datasets import load_dataset


from transformers import RobertaTokenizer
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# Load Dataset
model_name = 'roberta-base'
dataset = load_dataset("quora")
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Load validation dataset
sts_b = load_dataset("stsb_multi_mt", 'en')

# Define length for examples
max_sequence_length = 128
batch_size = 128
```

```{code-cell} ipython3

```

### Prepare Training TFRecords using Quora

* 1. Download Quora dataset.
* 2. We will take only those row where ```is_duplicate=True```. The model will be trained using ```in-batch``` negative loss.  
* 3. Example data looks like a pair of sentences
           ```sentence1 (left sentence): What is the best Android smartphone?,
              sentence2 (right sentence): What is the best Android smartphone ever?```

```{code-cell} ipython3
def parse_train(dataset, tokenizer, max_passage_length, key):
    """Function to parse examples which are is_duplicate=1

    Args:
        dataset (:obj:`dataet`): HF dataset
        tokenizer (:obj:`tokenizer`): HF Tokenizer
        max_passage_length (:obj:`int`): Passage Length
        key (:obj:`str`): Key of dataset (`train`, `validation` etc)
    """    
    result = {}
    for f in dataset[key]:
       
        question_left , question_right = f['questions']['text']
        question_left_input_ids =  tokenizer(question_left, max_length=max_passage_length, truncation=True)['input_ids'] 
        question_right_input_ids  =  tokenizer(question_right, max_length=max_passage_length, truncation=True)['input_ids']
        
        result = {}
        result['input_ids_left'] = question_left_input_ids
        result['input_ids_right'] = question_right_input_ids
        
        yield result
        
# Write using TF Writer
schema = {
    "input_ids_left": ("var_len", "int"),
    "input_ids_right": ("var_len", "int")
    
}

tfrecord_train_dir = tempfile.mkdtemp()
tfrecord_filename = 'quora'

tfwriter = TFWriter(schema=schema, 
                    file_name=tfrecord_filename, 
                    model_dir=tfrecord_train_dir,
                    tag='train',
                    overwrite=True
                    )

# Train dataset
train_parser_fn = parse_train(dataset, tokenizer, max_sequence_length, key='train')
tfwriter.process(parse_fn=train_parser_fn)
```

```{code-cell} ipython3

```

### Prepare Validation TFRecords using STS-b

1. Download STS dataset.
2. We will use this dataset to measure sentence embeddings by measuring the correlation

```{code-cell} ipython3
def parse_dev(dataset, tokenizer, max_passage_length, key):
    """Function to parse examples which are is_duplicate=1

    Args:
        dataset (:obj:`dataet`): HF dataset
        tokenizer (:obj:`tokenizer`): HF Tokenizer
        max_passage_length (:obj:`int`): Passage Length
        key (:obj:`str`): Key of dataset (`train`, `validation` etc)
    """    
    result = {}
    max_score = 5.0
    min_score = 0.0
    for f in dataset[key]:
        
        question_left = f['sentence1']
        question_right = f['sentence2']
        question_left_input_ids =  tokenizer(question_left, max_length=max_passage_length, truncation=True)['input_ids'] 
        question_right_input_ids  =  tokenizer(question_right, max_length=max_passage_length, truncation=True)['input_ids']
        
        result = {}
        result['input_ids_left'] = question_left_input_ids
        result['input_ids_right'] = question_right_input_ids
        score = f['similarity_score']
        # Normalize scores
        result['score'] = (score - min_score) / (max_score - min_score)
        yield result
        
# Write using TF Writer
schema = {
    "input_ids_left": ("var_len", "int"),
    "input_ids_right": ("var_len", "int"),
    "score": ("var_len", "float")
    
}

tfrecord_validation_dir = tempfile.mkdtemp()
tfrecord_validation_filename = 'sts'

tfwriter = TFWriter(schema=schema, 
                    file_name=tfrecord_validation_filename, 
                    model_dir=tfrecord_validation_dir,
                    tag='eval',
                    overwrite=True
                    )

# Train dataset
dev_parser_fn = parse_dev(sts_b, tokenizer, max_sequence_length, key='dev')
tfwriter.process(parse_fn=dev_parser_fn)
```

```{code-cell} ipython3

```

### Prepare  Training and Validation Dataset from TFRecords

```{code-cell} ipython3
# Read TFRecord

def add_mask_type_ids(item):
    
    item['input_mask_left'] = tf.ones_like(item['input_ids_left'])
    item['input_type_ids_left']= tf.zeros_like(item['input_ids_left'])
    item['input_mask_right'] = tf.ones_like(item['input_ids_right'])
    item['input_type_ids_right']= tf.zeros_like(item['input_ids_right'])
    
    labels = {}
    if 'score' in item:
        labels = {'score': item['score']}
        del item['score']
    
    return item, labels

# Train dataset
schema = json.load(open("{}/schema.json".format(tfrecord_train_dir)))
total_train_examples = json.load(open("{}/stats.json".format(tfrecord_train_dir)))['total_records']


all_files = tf.io.gfile.glob("{}/*.tfrecord".format(tfrecord_train_dir))
tf_reader = TFReader(schema=schema, 
                    tfrecord_files=all_files)

x_keys = ['input_ids_left', 'input_ids_right']
train_dataset = tf_reader.read_record(auto_batch=False, 
                                   keys=x_keys,
                                   batch_size=batch_size, 
                                   x_keys = x_keys, 
                                   shuffle=True
                                  )
train_dataset = train_dataset.map(add_mask_type_ids, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size, drop_remainder=True)


# Validation dataset
val_schema = json.load(open("{}/schema.json".format(tfrecord_validation_dir)))
all_val_files = tf.io.gfile.glob("{}/*.tfrecord".format(tfrecord_validation_dir))
tf_reader_val = TFReader(schema=val_schema, 
                    tfrecord_files=all_val_files)

x_keys_val = ['input_ids_left', 'input_ids_right', 'score']
validation_dataset = tf_reader_val.read_record(auto_batch=False, 
                                   keys=x_keys_val,
                                   batch_size=batch_size, 
                                   x_keys = x_keys_val, 
                                   shuffle=True
                                  )

# Static shapes makes things faster inside tf.function
# Especially for validation as we are passing batch examples to tf.function
padded_shapes = ({'input_ids_left': [max_sequence_length,], 
                 'input_mask_left':[max_sequence_length,],
                 'input_type_ids_left':[max_sequence_length,],
                 'input_ids_right': [max_sequence_length,],
                 'input_mask_right': [max_sequence_length,],
                 'input_type_ids_right': [max_sequence_length,]
                }, 
                 {'score': [None,]})
validation_dataset = validation_dataset.map(add_mask_type_ids,
                                            num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size,
                                                                                              drop_remainder=False,
                                                                                              padded_shapes=padded_shapes
                                                                                              )
```

```{code-cell} ipython3

```

### Build Sentence Transformer Model

```{code-cell} ipython3
import tensorflow as tf
from tf_transformers.core import LegacyLayer, LegacyModel


class Sentence_Embedding_Model(LegacyLayer):
    def __init__(
        self,
        model,
        is_training=False,
        use_dropout=False,
        **kwargs,
    ):
        r"""
        Simple Sentence Embedding using Keras Layer

        Args:
            model (:obj:`LegacyLayer/LegacyModel`):
                Model.
                Eg:`~tf_transformers.model.BertModel`.
            is_training (:obj:`bool`, `optional`, defaults to False): To train
            use_dropout (:obj:`bool`, `optional`, defaults to False): Use dropout
            use_bias (:obj:`bool`, `optional`, defaults to True): use bias
        """
        super(Sentence_Embedding_Model, self).__init__(
            is_training=is_training, use_dropout=use_dropout, name=model.name, **kwargs
        )

        self.model = model
        if isinstance(model, LegacyModel):
            self.model_config = model.model_config
        elif isinstance(model, tf.keras.layers.Layer):
            self.model_config = model._config_dict
        self._is_training = is_training
        self._use_dropout = use_dropout

        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)
        
    def get_mean_embeddings(self, token_embeddings, input_mask):
        """
        Mean embeddings
        """
        cls_embeddings = token_embeddings[:, 0, :] # 0 is CLS (<s>)
        # mask PAD tokens
        token_emb_masked = token_embeddings * tf.cast(tf.expand_dims(input_mask, 2), tf.float32)
        total_non_padded_tokens_per_batch = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.float32)
        # Convert to 2D
        total_non_padded_tokens_per_batch = tf.expand_dims(total_non_padded_tokens_per_batch, 1)
        mean_embeddings = tf.reduce_sum(token_emb_masked, axis=1)/ total_non_padded_tokens_per_batch
        return mean_embeddings

    def call(self, inputs):
        """Call"""
        
        # Extract left and right input pairs
        left_inputs = {k.replace('_left', ''):v for k,v in inputs.items() if 'left' in k}
        right_inputs = {k.replace('_right', ''):v for k,v in inputs.items() if 'right' in k}
        model_outputs_left = self.model(left_inputs)
        model_outputs_right = self.model(right_inputs)
        
        left_cls = model_outputs_left['cls_output']
        right_cls = model_outputs_right['cls_output']        

        left_mean_embeddings  = self.get_mean_embeddings(model_outputs_left['token_embeddings'], left_inputs['input_mask'])
        right_mean_embeddings  = self.get_mean_embeddings(model_outputs_right['token_embeddings'], right_inputs['input_mask'])
        
        cls_logits = tf.matmul(left_cls, right_cls, transpose_b=True)
        mean_logits = tf.matmul(left_mean_embeddings, right_mean_embeddings, transpose_b=True)
        
        
        results = {'left_cls_output': left_cls, 
                   'right_cls_output': right_cls, 
                   'left_mean_embeddings': left_mean_embeddings,
                   'right_mean_embeddings': right_mean_embeddings,
                   'cls_logits': cls_logits, 
                   'mean_logits': mean_logits}
        
        return results
        

    def get_model(self, initialize_only=False):
        """Get model"""
        inputs = self.model.input
        # Left and Right inputs
        main_inputs = {}
        for k, v in inputs.items():
            shape = v.shape
            main_inputs[k+'_left'] = tf.keras.layers.Input(
                            shape[1:], batch_size=v.shape[0], name=k+'_left', dtype=v.dtype
                        )
            
        for k, v in inputs.items():
            shape = v.shape
            main_inputs[k+'_right'] = tf.keras.layers.Input(
                            shape[1:], batch_size=v.shape[0], name=k+'_right', dtype=v.dtype
                        )        
        layer_outputs = self(main_inputs)
        if initialize_only:
            return main_inputs, layer_outputs
        model = LegacyModel(inputs=main_inputs, outputs=layer_outputs, name="sentence_embedding_model")
        model.model_config = self.model_config
        return model
```

```{code-cell} ipython3

```

### Load Model, Optimizer , Trainer

Our Trainer expects ```model```, ```optimizer``` and ```loss``` to be a function.

* 1. We will use ```Roberta``` as the base model and pass it to ```Sentence_Embedding_Model```, layer we built
* 2. We will use ```in-batch``` loss as the loss function, where every diagonal entry in the output is positive
and rest is negative

```{code-cell} ipython3
# Load Model
def get_model(model_name, is_training, use_dropout):
    """Get Model"""
    def model_fn():
        model = RobertaModel.from_pretrained(model_name)
        sentence_transformers_model = Sentence_Embedding_Model(model)
        sentence_transformers_model = sentence_transformers_model.get_model()
        return sentence_transformers_model
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

# Create loss
def in_batch_negative_loss():
    
    def loss_fn(y_true_dict, y_pred_dict):
        
        labels = tf.range(y_pred_dict['cls_logits'].shape[0])
        cls_loss  = cross_entropy_loss_for_classification(labels=labels, logits=y_pred_dict['cls_logits'])
        mean_loss = cross_entropy_loss_for_classification(labels=labels, logits=y_pred_dict['mean_logits'])
        
        result = {}
        result['cls_loss'] = cls_loss
        result['mean_loss'] = mean_loss
        result['loss'] = (cls_loss + mean_loss)/2.0
        return result
    
    return loss_fn
```

```{code-cell} ipython3

```

### Wandb Configuration

```{code-cell} ipython3
project = "TUTORIALS"
display_name = "roberta_quora_sentence_embedding"
wandb.init(project=project, name=display_name)
```

### Zero-Shot on STS before Training

* 1. Lets evaluate how good ```Roberta``` is to capture sentence embeddings before ```fine-tuning``` with Quora.
* 2. This gives us an indication whether the model is learning something or not on downstream fine-tuning.
* 3. We use ```CLS_OUTPUT```, pooler output of ```Roberta``` model as sentence embedding and evaluate using
```pearson``` and ```spearman``` correlation.

```{code-cell} ipython3
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr

model = RobertaModel.from_pretrained(model_name)

sentence1_embeddings = []
sentence2_embeddings = []
sts_labels = []
for batch_inputs, batch_labels in tqdm.tqdm(validation_dataset):
    left_inputs = {k.replace('_left', ''):v for k,v in batch_inputs.items() if 'left' in k}
    right_inputs = {k.replace('_right', ''):v for k,v in batch_inputs.items() if 'right' in k}
    left_outputs = model(left_inputs)
    right_outputs = model(right_inputs)
    
    # sentence 1 embeddings
    sentence1_embeddings.append(left_outputs['cls_output'])
    # sentence 2 embeddings
    sentence2_embeddings.append(right_outputs['cls_output'])
    sts_labels.append(batch_labels['score'])
    
sts_labels = tf.squeeze(tf.concat(sts_labels, axis=0), axis=1)
sentence1_embeddings = tf.concat(sentence1_embeddings, axis=0)
sentence2_embeddings = tf.concat(sentence2_embeddings, axis=0)

cosine_scores = 1 - (paired_cosine_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy()))
manhattan_distances = -paired_manhattan_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())
euclidean_distances = -paired_euclidean_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())
dot_products        = [np.dot(emb1, emb2) for emb1, emb2 in zip(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())]


eval_pearson_cosine, _    = pearsonr(sts_labels, cosine_scores)
eval_spearman_cosine, _   = spearmanr(sts_labels, cosine_scores)

eval_pearson_manhattan, _  = pearsonr(sts_labels, manhattan_distances)
eval_spearman_manhattan, _ = spearmanr(sts_labels, manhattan_distances)

eval_pearson_euclidean, _  = pearsonr(sts_labels, euclidean_distances)
eval_spearman_euclidean, _ = spearmanr(sts_labels, euclidean_distances)

eval_pearson_dot, _  = pearsonr(sts_labels, dot_products)
eval_spearman_dot, _ = spearmanr(sts_labels, dot_products)


print("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
    eval_pearson_cosine, eval_spearman_cosine))
print("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
    eval_pearson_manhattan, eval_spearman_manhattan))
print("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
    eval_pearson_euclidean, eval_spearman_euclidean))
print("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
    eval_pearson_dot, eval_spearman_dot))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr

class STSEvaluationCallback:
    def __init__(self) -> None:
        pass

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
            s1_cls = model_outputs['left_cls_output']
            s2_cls = model_outputs['right_cls_output']
            
            s1_mean = model_outputs['left_mean_embeddings']
            s2_mean = model_outputs['right_mean_embeddings']
            return s1_cls, s2_cls, s1_mean, s2_mean, batch_labels['score']
        
        S1_cls = []
        S2_cls = []
        S1_mean = []
        S2_mean = []
        sts_labels = []
        # This is a hack to make tqdm to print colour bar
        # TODO: fix it .
        pbar = tqdm.trange(validation_steps, colour="magenta", unit="batch")
        for step_counter in pbar:
            dist_inputs = next(validation_dataset_distributed)
            s1_cls, s2_cls, s1_mean, s2_mean, batch_scores = strategy.run(
                validate_run, args=(dist_inputs,)
            )
            s1_cls = tf.concat(
                trainer.distribution_strategy.experimental_local_results(s1_cls),
                axis=0,
            )
            s2_cls = tf.concat(
                            trainer.distribution_strategy.experimental_local_results(s2_cls),
                            axis=0,
                        )
            s1_mean = tf.concat(
                            trainer.distribution_strategy.experimental_local_results(s1_mean),
                            axis=0,
                        )
            s2_mean = tf.concat(
                                trainer.distribution_strategy.experimental_local_results(s2_mean),
                                        axis=0,
                                    )
            
            scores = tf.concat(
                trainer.distribution_strategy.experimental_local_results(
                    batch_scores
                ),
                axis=0,
            )

            S1_cls.append(s1_cls)
            S2_cls.append(s2_cls)
            S1_mean.append(s1_mean)
            S2_mean.append(s2_mean)
            sts_labels.append(scores)
            pbar.set_description(
                "Callback: Epoch {}/{} --- Step {}/{} ".format(
                    epoch, epochs, step_counter, validation_steps
                )
            )
            
            
        sts_labels = tf.squeeze(tf.concat(sts_labels, axis=0), axis=1)
        sentence1_embeddings = tf.concat(S1_cls, axis=0)
        sentence2_embeddings = tf.concat(S2_cls, axis=0)

        cosine_scores = 1 - (paired_cosine_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy()))
        manhattan_distances = -paired_manhattan_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())
        euclidean_distances = -paired_euclidean_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())
        dot_products        = [np.dot(emb1, emb2) for emb1, emb2 in zip(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())]


        eval_pearson_cosine, _    = pearsonr(sts_labels, cosine_scores)
        eval_spearman_cosine, _   = spearmanr(sts_labels, cosine_scores)

        eval_pearson_manhattan, _  = pearsonr(sts_labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(sts_labels, manhattan_distances)

        eval_pearson_euclidean, _  = pearsonr(sts_labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(sts_labels, euclidean_distances)

        eval_pearson_dot, _  = pearsonr(sts_labels, dot_products)
        eval_spearman_dot, _ = spearmanr(sts_labels, dot_products)

        metrics_result = {'pearson_cosine_cls': eval_pearson_cosine,
                          'spearman_cosine_cls': eval_spearman_cosine,
                          'pearson_manhattan_cls': eval_pearson_manhattan, 
                          'spearman_manhattan_cls': eval_spearman_manhattan, 
                          'pearson_euclidean_cls': eval_pearson_euclidean, 
                          'spearman_euclidean_cls': eval_spearman_euclidean, 
                          'pearson_dot_cls': eval_pearson_dot, 
                          'spearman_dot_cls': eval_spearman_dot}
        
        sentence1_embeddings = tf.concat(S1_mean, axis=0)
        sentence2_embeddings = tf.concat(S2_mean, axis=0)

        cosine_scores = 1 - (paired_cosine_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy()))
        manhattan_distances = -paired_manhattan_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())
        euclidean_distances = -paired_euclidean_distances(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())
        dot_products        = [np.dot(emb1, emb2) for emb1, emb2 in zip(sentence1_embeddings.numpy(), sentence2_embeddings.numpy())]


        eval_pearson_cosine, _    = pearsonr(sts_labels, cosine_scores)
        eval_spearman_cosine, _   = spearmanr(sts_labels, cosine_scores)

        eval_pearson_manhattan, _  = pearsonr(sts_labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(sts_labels, manhattan_distances)

        eval_pearson_euclidean, _  = pearsonr(sts_labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(sts_labels, euclidean_distances)

        eval_pearson_dot, _  = pearsonr(sts_labels, dot_products)
        eval_spearman_dot, _ = spearmanr(sts_labels, dot_products)
        
        metrics_result_mean = {'pearson_cosine_mean': eval_pearson_cosine,
                          'spearman_cosine_mean': eval_spearman_cosine,
                          'pearson_manhattan_mean': eval_pearson_manhattan, 
                          'spearman_manhattan_mean': eval_spearman_manhattan, 
                          'pearson_euclidean_mean': eval_pearson_euclidean, 
                          'spearman_euclidean_mean': eval_spearman_euclidean, 
                          'pearson_dot_mean': eval_pearson_dot, 
                          'spearman_dot_mean': eval_spearman_dot}
        
        metrics_result.update(metrics_result_mean)
        pbar.set_postfix(**metrics_result)
        
        if wandb:
            wandb.log(metrics_result, step=step)

        return metrics_result
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
epochs = 3
model_checkpoint_dir = 'MODELS/roberta_quora_embeddings'


# Total train examples
steps_per_epoch = total_train_examples // batch_size

# model
model_fn =  get_model(model_name, is_training=True, use_dropout=True)
# optimizer
optimizer_fn = get_optimizer(learning_rate, total_train_examples, batch_size, epochs)
# trainer (multi gpu strategy)
trainer = get_trainer(distribution_strategy='mirrored', num_gpus=2)
# loss
loss_fn = in_batch_negative_loss()
```

```{code-cell} ipython3

```

### Train :-)

* 1. Loss is coming down in epoch 1 itself.
* 2. Zershot evaluation after ```epoch 1``` shows that, ```pearson``` and ```spearman``` correlation increases to
```0.80```, which is significant improvemnet over ```Roberta``` base model, where we got ```0.43```.
* 3. Without training on ```STS-B```, we got a good evaluation score on ```STS-B dev``` using Zeroshot.

```{code-cell} ipython3
sts_callback = STSEvaluationCallback()
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
    training_loss_names = ['cls_loss', 'mean_loss'],
    validation_loss_names = ['cls_loss', 'mean_loss'],
    steps_per_call=10,
    callbacks=[sts_callback],
    wandb=wandb
)
```

```{code-cell} ipython3

```

### Visualize the Tensorboard

```{code-cell} ipython3
%load_ext tensorboard

%tensorboard --logdir MODELS/roberta_quora_embeddings/logs
```

```{code-cell} ipython3

```

### Load Trained Model for Testing and Save it as serialzed model

* 1. To get good sentence embedding , we need only ```Roberta``` model, which has been used as the ```base``` for
```Sentence_Embedding_Model``` .

```{code-cell} ipython3
# Save serialized version of the model

# Note: Ignore checkpoint warnings, it is because we save optimizer with checkpoint
# while we restoring, we take only model.


model_fn =  get_model(model_name, is_training=False, use_dropout=False)
model = model_fn()
model.load_checkpoint(model_checkpoint_dir)

# Roberta base (model.layers[-1] is Sentence_Embedding_Model )
model = model.layers[-1].model
model.save_transformers_serialized('{}/saved_model/'.format(model_checkpoint_dir))
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

### Model Serialization (Production)

* 1. Lets see how we can use this model to extract sentence embeddings
* 2. Print top K similar sentences from our embeddings from Quora Dataset

```{code-cell} ipython3
# Load serialized model

loaded = tf.saved_model.load("{}/saved_model/".format(model_checkpoint_dir))
model = loaded.signatures['serving_default']
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# Take 100000 sentences from Quora and calculate embeddings of that
quora_questions = []
for item in dataset['train']:
    quora_questions.extend(item['questions']['text'])
    
quora_questions = list(set(quora_questions))
quora_questions = quora_questions[:100000] # Take 100000
print("Total sentences {}".format(len(quora_questions)))

# Prepare Dataset
quora_dataset = tf.data.Dataset.from_tensor_slices({'questions': quora_questions})
quora_dataset = quora_dataset.batch(batch_size, drop_remainder=False)
```

```{code-cell} ipython3

```

### Quora Sentence Embeddings

```{code-cell} ipython3
quora_sentence_embeddings = []
for batch_questions in tqdm.tqdm(quora_dataset):
    batch_questions = batch_questions['questions'].numpy().tolist()
    batch_questions = [q.decode() for q in batch_questions]
    
    # Tokenize
    quora_inputs = tokenizer(batch_questions, max_length=max_sequence_length, padding=True, truncation=True, return_tensors='tf')
    quora_inputs['input_mask'] = quora_inputs['attention_mask']
    quora_inputs['input_type_ids'] = tf.zeros_like(quora_inputs['input_ids'])
    del quora_inputs['attention_mask'] # we dont want this

    model_outputs = model(**quora_inputs)
    quora_sentence_embeddings.append(model_outputs['cls_output'])
    
# Pack and Normalize
quora_sentence_embeddings = tf.nn.l2_normalize(tf.concat(quora_sentence_embeddings, axis=0), axis=-1)
```

```{code-cell} ipython3

```

### Most Similar Sentences

```{code-cell} ipython3
def most_similar(input_question, top_k=10):
    quora_inputs = tokenizer([input_question], max_length=max_sequence_length, padding=True, truncation=True, return_tensors='tf')
    quora_inputs['input_mask'] = quora_inputs['attention_mask']
    quora_inputs['input_type_ids'] = tf.zeros_like(quora_inputs['input_ids'])
    del quora_inputs['attention_mask'] # we dont want this
    model_outputs = model(**quora_inputs)
    query_vector = model_outputs['cls_output']
    query_vector = tf.nn.l2_normalize(query_vector, axis=1)

    scores = tf.matmul(query_vector, quora_sentence_embeddings, transpose_b=True)
    top_k_values = tf.nn.top_k(scores, k=top_k)
    for i in range(top_k):
        best_index = top_k_values.indices.numpy()[0][i]
        best_prob = top_k_values.values.numpy()[0][i]
        print(quora_questions[best_index], '-->', best_prob)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
input_question = 'What is the best way to propose a girl?'
most_similar(input_question)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
input_question = 'How can I start learning Deep Learning?'
most_similar(input_question)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
input_question = 'Best tourist destinations in India'
most_similar(input_question)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
input_question = 'Why classical music is so relaxing?'
most_similar(input_question)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
