# coding=utf-8
# Copyright 2020 The tf_transformers Authors.
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

import collections
import json
import os
import random

import six
import tensorflow as tf
from absl import logging

from tf_transformers.data import separate_x_y

logging.set_verbosity("INFO")

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    if isinstance(value, list):
        value = [six.ensure_binary(token) for token in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        # value = str([six.ensure_text(token, "utf-8") for \
        # token in value]).encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(values):
    if isinstance(values, int):
        values = [values]
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


TF_SCHEMA = {"var_len": tf.io.VarLenFeature, "fixed_len": tf.io.FixedLenFeature}

TF_VALUE = {"bytes": tf.string, "int": tf.int64, "float": tf.float32}

TF_FUNC = {"bytes": _bytes_feature, "int": _int_feature, "float": _float_feature}


class TFWriter(object):
    """TFWriter class . This class is responsible
    to write tfrecords, based on given schema and data.
    """

    def __init__(
        self,
        schema,
        file_name,
        model_dir=None,
        tag="dev",
        n_files=10,
        shuffle=True,
        max_files_per_record=10000,
        overwrite=False,
        verbose_counter=1000,
    ):
        """
        Args:
            schema: dict - (this is where schema of the tfrecords specified)
            file_name: str - file name
            model_dir: str - TFRecords will write to this
                             model dir . If not given, use the default directory
            tag: str - 'train' or 'dev'
            n_files: int - If `tag` == 'train': file will be
                     split into `n_fles` for randomness
            shuffle: bool
            max_files_per_record: No of individual files
                    (can be a sentence/ tokenized sentence) per record
            overwrite: bool - If True, we will overwrite
                    tfrecords of the same name

        Raises:
            Error if the model_dir / the file exists .
            You can pass overwrite = True to disable this behaviour

        """

        self.shuffle = shuffle
        self.max_files_per_record = max_files_per_record
        # Schema Check
        self.is_schema_valid(schema)
        self.tag = tag

        if tag not in ["train", "dev"]:
            logging.info("Unknown tag {} found".format(tag))
            raise Exception("Unknwon Tag")

        def is_check(all_files):
            for file_ in all_files:
                if os.path.exists(file_):
                    logging.info(
                        "File exists, overwrite is not recommended. \
                        If you want to overwrite, pass `overwrite`=True"
                    )
                    raise FileExistsError(file_)

        # we need this file to write the schema to the model_dir
        schema_file_name = "schema.json"
        if model_dir:
            if overwrite is False:
                if os.path.exists(model_dir):
                    logging.info("Model directory {} exists".format(model_dir))
                    raise FileExistsError(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            self.file_name = file_name = os.path.join(model_dir, file_name.replace(".tfrecord", ""))
            schema_file_name = os.path.join(model_dir, schema_file_name)

        if tag == "train":
            if self.shuffle:
                self.all_files = ["{}_{}_{}_{}.tfrecord".format(file_name, tag, i, n_files) for i in range(n_files)]
                self.examples_per_record = {file_: 0 for file_ in self.all_files}
                if overwrite is False:
                    is_check(self.all_files)
                self.all_writer = [tf.io.TFRecordWriter(file_) for file_ in self.all_files]
            else:
                self.current_writer = 0
                self.temp_writers = []
                self.examples_per_record = {}
                self.current_file_name = "{}_{}_{}.tfrecord".format(self.file_name, self.tag, self.current_writer)
                self.examples_per_record[self.current_file_name] = 0
                self.current_file = tf.io.TFRecordWriter(self.current_file_name)
                self.temp_writers.append(self.current_file)

        else:
            n_files = 1
            self.all_files = ["{}_{}_{}_{}.tfrecord".format(file_name, i, tag, n_files) for i in range(n_files)]
            if overwrite is False:
                is_check(self.all_files)
            self.all_writer = [tf.io.TFRecordWriter(file_) for file_ in self.all_files]

        self.schema = schema
        self.schema_writer_fn = self.generate_schema_from_dict(schema)

        self.verbose_counter = verbose_counter
        self.global_counter = 0

        # Save schema for further reading
        with open(schema_file_name, "w") as f:
            json.dump(schema, f, indent=2)

    def is_schema_valid(self, schema):
        """
        simple schema validation check
        """
        for k, v in schema.items():
            if v[0] == "var_len":
                assert len(v) == 2
                assert v[1] in TF_VALUE

            if v[0] == "fixed_len":
                assert len(v) == 3
                assert v[1] in TF_VALUE
                assert isinstance(v[2], list)

    def close_sess(self):
        if self.shuffle:
            for file_writer in self.all_writer:
                file_writer.close()
            logging.info("All writer objects closed")
        else:
            for file_writer in self.temp_writers:
                file_writer.close()
            logging.info("All writer objects closed")

    def generate_schema_from_dict(self, schema_dict):
        """
        schema_dict: a dict
        """
        allowed_schema_types = ["var_len", "fixed_len"]
        allowed_schema_values = ["bytes", "int", "float"]

        def check_schema(schema_dict):
            for _, value in schema_dict.items():
                schema_key = value[0]
                schema_value = value[1]
                if schema_key not in allowed_schema_types:
                    error_message = "{} not in {}".format(schema_key, allowed_schema_types)
                    raise ValueError(error_message)
                if schema_value not in allowed_schema_values:
                    error_message = "{} not in {}".format(schema_value, allowed_schema_values)
                    raise ValueError(error_message)

        check_schema(schema_dict)

        schema_writer_dict = {}
        for key, value in schema_dict.items():
            schema_writer_dict[key] = TF_FUNC[value[1]]  # _bytes_feature
        return schema_writer_dict

    def process(self, parse_fn):
        """This function will iterate over parse_fn and keep writing it TFRecord"""
        """
        parse_fn: function which should be an iterator or generator
        """
        if hasattr(parse_fn, "__iter__") and not hasattr(parse_fn, "__len__"):
            for entry in parse_fn:
                self.write_record(entry)
            logging.info("Total individual observations/examples written is {}".format(self.global_counter))
            self.close_sess()
        else:
            raise ValueError("Expected `parse_fn` to be a generator/iterator ")

    def write_record(self, input):
        """Writes a input to a TFRecord example."""
        """
        input: dict (dict of key, elem to write to tf-record)
        """
        features = collections.OrderedDict()
        for key, value in input.items():
            if self.schema[key][0] == "fixed_len":
                if self.schema[key][2] != []:
                    shape = self.schema[key][2][0]
                    if len(value) != shape:
                        raise ValueError(
                            "`{}` has schema shape `{}`, but provided \
                              values `{}` has shape `{}`".format(
                                key, shape, value, len(value)  # noqa
                            )
                        )

            if isinstance(value, six.text_type):
                value = six.ensure_binary(value, "utf-8")
            features[key] = self.schema_writer_fn[key](value)
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))

        if self.tag == "train":
            if self.shuffle:
                index = random.choice(range(len(self.all_writer)))
                the_writer = self.all_writer[index]
                the_writer.write(example_proto.SerializeToString())
                self.examples_per_record[self.all_files[index]] += 1
                self.global_counter += 1
            else:

                # If global counter(no of individual records processed)
                # exceeds max_files_per_record then increment self.current_writer
                if self.global_counter > (self.current_writer + 1) * self.max_files_per_record:
                    self.current_writer += 1
                    self.current_file_name = "{}_{}_{}.tfrecord".format(self.file_name, self.tag, self.current_writer)
                    self.examples_per_record[self.current_file_name] = 0
                    self.current_file = tf.io.TFRecordWriter(self.current_file_name)
                    self.temp_writers.append(self.current_file)

                the_writer = self.current_file
                the_writer.write(example_proto.SerializeToString())
                self.examples_per_record[self.current_file_name] += 1
                self.global_counter += 1

            if self.global_counter % self.verbose_counter == 0:
                logging.info("Wrote {} tfrecods".format(self.global_counter))
        else:
            the_writer = self.all_writer[0]
            the_writer.write(example_proto.SerializeToString())
            self.global_counter += 1

            if self.global_counter % self.verbose_counter == 0:
                logging.info("Wrote {} tfrecods".format(self.global_counter))


class TFReader(object):
    """
    TFReader class . This class is responsible
    to read tfrecords, based on given schema.

    """

    def __init__(self, schema, tfrecord_files, shuffle_files=False, keys=[]):

        if not isinstance(tfrecord_files, (list, tuple)):
            raise Exception("input must be a list or tuple of files")
        self.schema = schema
        self.tfrecord_files = tfrecord_files
        self.shuffle_files = shuffle_files
        self.keys = keys
        if self.keys == []:
            self.keys = self.schema.keys()
        self.schema_reader_fn, self.schema_writer_fn = self.generate_schema_from_dict(schema)

    def generate_schema_from_dict(self, schema_dict):
        """
        schema_dict: a dict
        """
        allowed_schema_types = ["var_len", "fixed_len"]
        allowed_schema_values = ["bytes", "int", "float"]

        def check_schema(schema_dict):
            for _, value in schema_dict.items():
                schema_key = value[0]
                schema_value = value[1]
                if schema_key not in allowed_schema_types:
                    error_message = "{} not in {}".format(schema_key, allowed_schema_types)
                    raise ValueError(error_message)
                if schema_value not in allowed_schema_values:
                    error_message = "{} not in {}".format(schema_value, allowed_schema_values)
                    raise ValueError(error_message)

        check_schema(schema_dict)

        # Schema reader function is here

        schema_reader_dict = {}
        for key, value in schema_dict.items():
            if self.keys and key not in self.keys:
                continue

            if value[0] == "var_len":
                schema_reader_dict[key] = tf.io.VarLenFeature(TF_VALUE[value[1]])
            if value[0] == "fixed_len":
                # Fixed len should have shape mentioned in the schema
                shape = value[2]
                schema_reader_dict[key] = tf.io.FixedLenFeature(
                    shape=shape, dtype=TF_VALUE[value[1]], default_value=None
                )

        schema_writer_dict = {}
        for key, value in schema_dict.items():
            schema_writer_dict[key] = TF_FUNC[value[1]]  # _bytes_feature
        return schema_reader_dict, schema_writer_dict

    def decode_record_var(self, record, keys=[]):
        """Decodes a record to a TensorFlow example."""
        feature_dict = tf.io.parse_single_example(record, self.schema_reader_fn)

        parse_dict = feature_dict.copy()
        for k in self.keys:
            v = feature_dict[k]
            if v.dtype == tf.int64:
                v = tf.cast(v, tf.int32)
            if self.schema[k][0] == "var_len":
                parse_dict[k] = tf.sparse.to_dense(v)

        return parse_dict

    def auto_batch(
        self,
        tf_dataset,
        batch_size,
        padded_values=None,
        padded_shapes=None,
        x_keys=None,
        y_keys=None,
        shuffle=False,
        drop_remainder=False,
        shuffle_buffer_size=10000,
        prefetch_buffer_size=100,
    ):
        """Auto Batching

        Args:
            tf_dataset : TF dataset
            x_keys (optional): List of key names. We will filter based on this.
            y_keys (optional): List of key names.
            shuffle (bool, optional): [description]. Defaults to False.
            shuffle_buffer_size (int, optional): [description]. Defaults to 10000.

        Returns:
            batched tf dataset
        """
        element_spec = tf_dataset.element_spec
        _padded_values = {}
        if not padded_values:
            padded_values = {}
        # sometimes we might have to have sme custom values other than 0
        for k, v in element_spec.items():
            if k in padded_values:
                value = padded_values[k]
                _padded_values[k] = tf.constant(value, dtype=value.dtype)
            else:
                _padded_values[k] = tf.constant(0, dtype=v.dtype)
        dataset = tf_dataset.padded_batch(
            padding_values=_padded_values,
            padded_shapes=padded_shapes,
            batch_size=batch_size,
            drop_remainder=drop_remainder,
        )
        # fmt: off
        if x_keys and y_keys:
            dataset = dataset.map(lambda x: separate_x_y(x, x_keys, y_keys), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # noqa
        # fmt: on
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=None, reshuffle_each_iteration=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def read_record(self, keys=[], auto_batch=False, **kwargs):
        """Read TF records

        Args:
            keys (list, optional): List of keys to read from the records
            auto_batch (bool, optional): Whethe to auto batch data

        Returns:
            [type]: [description]
        """
        dataset = tf.data.Dataset.list_files(self.tfrecord_files, shuffle=self.shuffle_files)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=8,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        def decode_fn(record):
            return self.decode_record_var(record, keys)

        dataset = dataset.map(decode_fn)
        # Using `ignore_errors()` will drop the element that causes an error.
        dataset = dataset.apply(tf.data.experimental.ignore_errors())  # ==> {1., 0.5, 0.2}
        if auto_batch:
            dataset = self.auto_batch(dataset, **kwargs)
        return dataset


if __name__ == "__main__":

    # Dummy example

    tf_dummy_schema = {
        "correct": ("var_len", "bytes"),
        "bad": ("var_len", "bytes"),
        "correct_tokens": ("var_len", "bytes"),
        "bad_tokens": ("var_len", "bytes"),
        "correct_indexes": ("var_len", "int"),
        "bad_indexes": ("var_len", "int"),
        "source": ("var_len", "bytes"),
        "unique_id": ("var_len", "float"),
        "simple_float": ("var_len", "float"),
    }

    data_1 = {
        "correct": b"Hi how are you",
        "bad": b"Hi how are your?",
        "correct_tokens": [b"hi", b"how", b"are", b"you"],
        "bad_tokens": [b"hi", b"how", b"are", b"your?"],
        "correct_indexes": [1, 2, 3, 4, 5],
        "bad_indexes": [1, 2, 3, 4, 5],
        "source": b"dummy_source",
        "unique_id": 1230344.0,
        "simple_float": [0.1, 0.2, 0.3, 0.4],
    }

    tf_writer = TFWriter(tf_dummy_schema, "test.tfrecord", tag="train", overwrite=True)

    for i in range(10000):
        tf_writer.write_record(data_1)

    import glob

    all_files = glob.glob("test**")
    print("All files", all_files)
    tf_reader = TFReader(tf_dummy_schema, all_files)
    dataset = tf_reader.read_record()

    # Dummy example

    tf_dummy_schema = {
        "correct": ("var_len", "bytes"),
        "bad": ("var_len", "bytes"),
        "correct_tokens": ("var_len", "bytes"),
        "bad_tokens": ("fixed_len", "bytes", [4]),
        "correct_indexes": ("var_len", "int"),
        "bad_indexes": ("var_len", "int"),
        "source": ("fixed_len", "bytes", []),
        "unique_id": ("fixed_len", "float", []),
        "simple_float": ("fixed_len", "float", [4]),
    }

    data_1 = {
        "correct": b"Hi how are you",
        "bad": b"Hi how are your?",
        "correct_tokens": [b"hi", b"how", b"are", b"you"],
        "bad_tokens": [b"hi", b"how", b"your?", b":-)"],
        "correct_indexes": [1, 2, 3, 4, 5],
        "bad_indexes": [1, 2, 3, 4, 5],
        "source": b"dummy_source",
        "unique_id": 1230344.0,
        "simple_float": [0.1, 0.2, 0.3, 0.4],
    }

    data_2 = {
        "correct": b"Hi how are you",
        "bad": b"Hi how are your?",
        "correct_tokens": [b"hi", b"how", b"are", b"you"],
        "bad_tokens": [b"hi", b"how", b"your?", b"are"],
        "correct_indexes": [1, 2, 3, 4, 5],
        "bad_indexes": [1, 2, 3, 4, 5],
        "source": b"dummy_source",
        "unique_id": 1230344.0,
        "simple_float": [0.1, 0.2, 0.3, 0.4],
    }

    tf_writer = TFWriter(tf_dummy_schema, "test2.tfrecord", tag="train", overwrite=True)
    for i in range(1000):
        tf_writer.write_record(data_1)
        tf_writer.write_record(data_2)

    all_files = glob.glob("test2**")
    print("All files", all_files)
    tf_reader = TFReader(tf_dummy_schema, all_files, keys=[])
    dataset = tf_reader.read_record()
