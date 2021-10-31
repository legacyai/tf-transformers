# coding=utf-8
# Copyright 2021 TF-Transformers Authors.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Prepare TFRecords for Long Blocker model"""
import argparse
import glob
import json
import os

from datasets import load_dataset
from model import get_tokenizer

from tf_transformers.data import TFReader, TFWriter

encoder_max_seq_length = 4096
decoder_max_seq_length = 256


def write_tfrecord_t5(
    data, tokenizer, encoder_max_length, decoder_max_length, mode, tfrecord_dir, take_sample=False, verbose=10000
):

    if mode not in ["train", "eval"]:
        raise ValueError("Inavlid mode `{}` specified. Available mode is ['train', 'eval']".format(mode))

    def get_tfrecord_example(data):
        for f in data:
            inputs_hf = tokenizer('long summarize: ' + f['article'], truncation=True, max_length=encoder_max_length)

            input_ids = inputs_hf['input_ids'][:-1]  # skip sep
            input_mask = inputs_hf['attention_mask'][:-1]  # skip sep

            decoder_input_ids = tokenizer(f['abstract'], truncation=True, max_length=decoder_max_length)['input_ids']

            # T5 pad id is the decoder_start_token_id
            decoder_start_token_id = tokenizer.pad_token_id
            decoder_input_ids = [decoder_start_token_id] + decoder_input_ids
            # decoder_input_type_ids = [0] * len(decoder_input_ids)

            result = {}
            result['encoder_input_ids'] = input_ids
            result['encoder_input_mask'] = input_mask
            result['decoder_input_ids'] = decoder_input_ids[:-1]  # except last word

            result['labels'] = decoder_input_ids[1:]  # not including first word
            result['labels_mask'] = [1] * len(result['labels'])
            result['text'] = f["abstract"]
            # Decoder doesnt need input_mask because by default decoder has causal mask mode

            yield result

    schema = {
        "encoder_input_ids": ("var_len", "int"),
        "encoder_input_mask": ("var_len", "int"),
        "decoder_input_ids": ("var_len", "int"),
        "labels": ("var_len", "int"),
        "labels_mask": ("var_len", "int"),
        "text": ("var_len", "bytes"),
    }

    # Create a temp dir
    if mode == "train":
        # Write tf records
        train_data_dir = os.path.join(tfrecord_dir, "train")
        tfrecord_filename = 'pubmed'
        tfwriter = TFWriter(
            schema=schema,
            file_name=tfrecord_filename,
            model_dir=train_data_dir,
            tag='train',
            overwrite=True,
            verbose_counter=verbose,
        )
        data_train = data
        # Take sample
        if take_sample:
            data_train = data_train.select(range(500))

        tfwriter.process(parse_fn=get_tfrecord_example(data_train))
    if mode == "eval":
        # Write tfrecords
        eval_data_dir = os.path.join(tfrecord_dir, "eval")
        tfrecord_filename = 'pubmed'
        tfwriter = TFWriter(
            schema=schema,
            file_name=tfrecord_filename,
            model_dir=eval_data_dir,
            tag='eval',
            overwrite=True,
            verbose_counter=verbose,
        )
        data_eval = data
        # Take sample
        if take_sample:
            data_eval = data_eval.select(range(500))
        tfwriter.process(parse_fn=get_tfrecord_example(data_eval))


def read_tfrecord(tfrecord_dir, max_seq_length, batch_size, shuffle=False, drop_remainder=False):
    """Read TFRecords"""
    padded_shapes = {
        'encoder_input_ids': [
            max_seq_length,
        ],
        'encoder_input_mask': [
            max_seq_length,
        ],
        'decoder_input_ids': [
            None,
        ],
        'labels': [
            None,
        ],
        'labels_mask': [
            None,
        ],
    }
    # Read tfrecord to dataset
    schema = json.load(open("{}/schema.json".format(tfrecord_dir)))
    stats = json.load(open('{}/stats.json'.format(tfrecord_dir)))
    all_files = glob.glob("{}/*.tfrecord".format(tfrecord_dir))
    tf_reader = TFReader(schema=schema, tfrecord_files=all_files)

    x_keys = ['encoder_input_ids', 'encoder_input_mask', 'decoder_input_ids']
    y_keys = ['labels', 'labels_mask', 'text']
    dataset = tf_reader.read_record(
        auto_batch=True,
        keys=x_keys,
        padded_shapes=padded_shapes,
        batch_size=batch_size,
        x_keys=x_keys,
        y_keys=y_keys,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
    )
    return dataset, stats['total_records']


def write_tfrecord_bart(
    data, tokenizer, encoder_max_length, decoder_max_length, mode, tfrecord_dir, take_sample=False, verbose=10000
):

    if mode not in ["train", "eval"]:
        raise ValueError("Inavlid mode `{}` specified. Available mode is ['train', 'eval']".format(mode))

    def get_tfrecord_example(data):
        for f in data:
            inputs_hf = tokenizer('long summarize: ' + f['article'], truncation=True, max_length=encoder_max_length)

            input_ids = inputs_hf['input_ids'][:-1]  # skip sep
            input_mask = inputs_hf['attention_mask'][:-1]  # skip sep

            decoder_input_ids = tokenizer(f['abstract'], truncation=True, max_length=decoder_max_length)['input_ids']

            # Bart eos id is the decoder_start_token_id
            decoder_start_token_id = tokenizer.eos_token_id
            decoder_input_ids = [decoder_start_token_id] + decoder_input_ids
            # decoder_input_type_ids = [0] * len(decoder_input_ids)

            result = {}
            result['encoder_input_ids'] = input_ids
            result['encoder_input_mask'] = input_mask
            result['decoder_input_ids'] = decoder_input_ids[:-1]  # except last word

            result['labels'] = decoder_input_ids[1:]  # not including first word
            result['labels_mask'] = [1] * len(result['labels'])
            result['text'] = f["abstract"]

            # Decoder doesnt need input_mask because by default decoder has causal mask mode

            yield result

    schema = {
        "encoder_input_ids": ("var_len", "int"),
        "encoder_input_mask": ("var_len", "int"),
        "decoder_input_ids": ("var_len", "int"),
        "labels": ("var_len", "int"),
        "labels_mask": ("var_len", "int"),
        "text": ("var_len", "bytes"),
    }

    # Create a temp dir
    if mode == "train":
        # Write tf records
        train_data_dir = os.path.join(tfrecord_dir, "train")
        tfrecord_filename = 'pubmed'
        tfwriter = TFWriter(
            schema=schema,
            file_name=tfrecord_filename,
            model_dir=train_data_dir,
            tag='train',
            overwrite=True,
            verbose_counter=verbose,
        )
        data_train = data
        # Take sample
        if take_sample:
            data_train = data_train.select(range(500))

        tfwriter.process(parse_fn=get_tfrecord_example(data_train))
    if mode == "eval":
        # Write tfrecords
        eval_data_dir = os.path.join(tfrecord_dir, "eval")
        tfrecord_filename = 'pubmed'
        tfwriter = TFWriter(
            schema=schema,
            file_name=tfrecord_filename,
            model_dir=eval_data_dir,
            tag='eval',
            overwrite=True,
            verbose_counter=verbose,
        )
        data_eval = data
        # Take sample
        if take_sample:
            data_eval = data_eval.select(range(500))
        tfwriter.process(parse_fn=get_tfrecord_example(data_eval))


def main():
    """Prepare TFRecords"""

    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, metavar='', help="Name of the model")
    parser.add_argument('-t', '--tfrecord_dir', type=str, metavar='', help="Path of tfrecords")
    # Parse and print the results
    args = parser.parse_args()

    model_name = args.model_name
    tfrecord_dir = args.tfrecord_dir
    take_sample = False

    tokenizer = get_tokenizer(model_name)
    if model_name.startswith('bart') or model_name.startswith('facebook/bart'):
        write_tfrecord = write_tfrecord_bart
    elif model_name.startswith('t5'):
        write_tfrecord = write_tfrecord_t5
    else:
        raise ValueError("Unsupported model name {}".format(model_name))

    # Load dataset
    dataset = load_dataset("scientific_papers", "pubmed")
    # # Train Tfrecords
    write_tfrecord(
        dataset['train'],
        tokenizer,
        encoder_max_seq_length,
        decoder_max_seq_length,
        "train",
        tfrecord_dir,
        take_sample,
        verbose=1000,
    )

    # # Eval Tfrecords
    write_tfrecord(
        dataset['validation'],
        tokenizer,
        encoder_max_seq_length,
        decoder_max_seq_length,
        "eval",
        tfrecord_dir,
        take_sample,
        verbose=1000,
    )


if __name__ == "__main__":
    main()
