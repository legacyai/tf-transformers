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

```{code-cell}

```

# Writing and Reading TFRecords


Tensoflow-Transformers has off the shelf support to write and read tfrecord with so much ease.
It also allows you to shard, shuffle and batch your data most of the times, with minimal code.

Here we will see, how can we make use of these utilities to write and read tfrecords.

For this examples, we will be using a [**Squad Dataset**](https://huggingface.co/datasets/squad "Squad Dataset"), to convert it to a text to text problem using
GPT2 Tokenizer.

```{code-cell}

```

```{code-cell}
from tf_transformers.data import TFWriter, TFReader
from transformers import GPT2TokenizerFast

from datasets import load_dataset

import tempfile
import json
import glob
```

```{code-cell}

```

## Load Data and Tokenizer

We will load dataset and tokenizer. Then we will define the length for the examples.
It is important to make sure we have limit the length within the allowed limit of each models.

```{code-cell}

```

```{code-cell}
# Load Dataset
dataset = load_dataset("squad")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# Define length for examples
max_passage_length = 384
max_question_length = 64
max_answer_length = 40
```

```{code-cell}

```

## Write TFRecord

To write a TFRecord, we need to provide a schema (**dict**). This schema supports **int**, **float**, **bytes**.

**TFWriter**, support [**FixedLen**](https://www.tensorflow.org/api_docs/python/tf/io/FixedLenFeature) and
[**VarLen**](https://www.tensorflow.org/api_docs/python/tf/io/VarLenFeature) feature types. 

The recommended and easiest is to use **Varlen**, this will be faster and easy to write and read.
We can also pad it accordingly after reading.

```{code-cell}

```

```{code-cell}
def parse_train(dataset, tokenizer, max_passage_length, max_question_length, max_answer_length, key):
    """Function o to parse examples

    Args:
        dataset (:obj:`dataet`): HF dataset
        tokenizer (:obj:`tokenizer`): HF Tokenizer
        max_passage_length (:obj:`int`): Passage Length
        max_question_length (:obj:`int`): Question Length
        max_answer_length (:obj:`int`): Answer Length
        key (:obj:`str`): Key of dataset (`train`, `validation` etc)
    """    
    result = {}
    for f in dataset[key]:
        question_input_ids =  tokenizer(item['context'], max_length=max_passage_length, truncation=True)['input_ids'] + [tokenizer.bos_token_id]
        passage_input_ids  =  tokenizer(item['question'], max_length=max_question_length, truncation=True)['input_ids']  + \
        [tokenizer.bos_token_id] 
        
        # Input Question + Context
        # We should make sure that we will mask labels here,as we dont want model to predict inputs
        input_ids = question_input_ids + passage_input_ids
        labels_mask = [0] * len(input_ids)
        
        # Answer part
        answer_ids = tokenizer(item['answers']['text'][0], max_length=max_answer_length, truncation=True)['input_ids'] + \
        [tokenizer.bos_token_id]
        input_ids = input_ids + answer_ids
        labels_mask = labels_mask + [1] * len(answer_ids)
        
        # Shift positions to make proper training examples
        labels = input_ids[1:]
        labels_mask = labels_mask[1:]
        
        input_ids = input_ids[:-1]

        result = {}
        result['input_ids'] = input_ids
        
        result['labels'] = labels
        result['labels_mask'] = labels_mask
        
        yield result
        
# Write using TF Writer

schema = {
    "input_ids": ("var_len", "int"),
    "labels": ("var_len", "int"),
    "labels_mask": ("var_len", "int"),
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
train_parser_fn = parse_train(dataset, tokenizer, max_passage_length, max_question_length, max_answer_length, key='train')
tfwriter.process(parse_fn=train_parser_fn)
```

```{code-cell}

```

## Read TFRecords

To read a TFRecord, we need to provide a schema (**dict**). This schema supports **int**, **float**, **bytes**.

**TFWReader**, support [**FixedLen**](https://www.tensorflow.org/api_docs/python/tf/io/FixedLenFeature) and
[**VarLen**](https://www.tensorflow.org/api_docs/python/tf/io/VarLenFeature) feature types. 
We can also **auto_batch**, **shuffle**, choose the optional keys (not all keys in tfrecords) might not be required while reading, etc in a single function.

```{code-cell}

```

```{code-cell}
# Read TFRecord

schema = json.load(open("{}/schema.json".format(tfrecord_train_dir)))
all_files = glob.glob("{}/*.tfrecord".format(tfrecord_train_dir))
tf_reader = TFReader(schema=schema, 
                    tfrecord_files=all_files)

x_keys = ['input_ids']
y_keys = ['labels', 'labels_mask']
batch_size = 16
train_dataset = tf_reader.read_record(auto_batch=True, 
                                   keys=x_keys,
                                   batch_size=batch_size, 
                                   x_keys = x_keys, 
                                   y_keys = y_keys,
                                   shuffle=True, 
                                   drop_remainder=True
                                  )
```

```{code-cell}

```

```{code-cell}
for (batch_inputs, batch_labels) in train_dataset:
    print(batch_inputs, batch_labels)
    break
```

```{code-cell}

```

```{code-cell}

```
