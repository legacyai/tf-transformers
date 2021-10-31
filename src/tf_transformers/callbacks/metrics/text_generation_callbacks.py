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
"""A simple metric callback for some Text Generation metrics in Tensorflow 2.0"""

import tempfile

import pandas as pd
import tensorflow as tf
import tqdm
from absl import logging
from rouge_score import rouge_scorer, scoring

from tf_transformers.text import TextDecoder

_ALL_METRIC_NAMES = {'rouge': True}


class TextGenerationMetricCallback:
    def __init__(
        self,
        model,
        tokenizer,
        decoder_kwargs={"mode": "greedy", "max_iterations": 64, "eos_id": -100},
        decoder_start_token_id=None,
        input_mask_ids=-1,
        input_type_ids=-1,
        metric_name_list=('rouge',),
        validation_dataset: tf.data.Dataset = None,
    ) -> None:
        """

        Args:
            model: tf.keras.Model
            tokenizer: Huggingface tokenizer
            decoder_kwargs: Dict of kwargs
            decoder_start_token_id: int
            input_mask_ids: int
            input_type_ids: int
            metric_name_list: tuple
            validation_dataset (tf.data.Dataset, optional): Validation dataset
        """
        for metric_name in metric_name_list:
            if metric_name not in _ALL_METRIC_NAMES:
                raise ValueError(
                    "metric {} not found in supported metric list {}".format(metric_name, _ALL_METRIC_NAMES)
                )
        self.model = model
        self.tokenizer = tokenizer
        self.decoder_kwargs = decoder_kwargs
        self.metric_name_list = metric_name_list
        self.decoder_start_token_id = decoder_start_token_id
        self.input_mask_ids = input_mask_ids
        self.input_type_ids = input_type_ids
        self.validation_dataset = validation_dataset

    def __call__(self, traininer_kwargs):
        """This is getting called inside the trainer class"""
        logging.info("Callback for {} is in progress . . . . . . . . . .".format(self.metric_name_list))
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        # This is non distribute
        validation_dataset = traininer_kwargs['validation_dataset']
        model_checkpoint_dir = traininer_kwargs['model_checkpoint_dir']
        # No validation dataset has been provided
        if validation_dataset is None:
            if self.validation_dataset is None:
                raise ValueError(
                    "No validation dataset has been provided either in the trainer class, \
                                 or when callback is initialized. Please provide a validation dataset"
                )
            else:
                validation_dataset = self.validation_dataset

        # Model from trainer
        self.model.load_checkpoint(model_checkpoint_dir)
        dirpath = tempfile.mkdtemp()

        # save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
        self.model.save_transformers_serialized(dirpath, overwrite=True)

        # load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        loaded = tf.saved_model.load(dirpath)

        decoder = TextDecoder(
            model=loaded,
            decoder_start_token_id=self.decoder_start_token_id,
            input_type_ids=self.input_type_ids,
            input_mask_ids=self.input_mask_ids,
        )

        original_summaries = []
        predicted_summaries = []
        # Save model as saved_model and load it
        for dist_inputs in tqdm.tqdm(validation_dataset):
            batch_inputs, batch_labels = dist_inputs
            decoder_outputs = decoder.decode(batch_inputs, **self.decoder_kwargs)

            predicted_ids = decoder_outputs['predicted_ids']
            predicted_ids_sliced = []
            predicted_ids = predicted_ids[:, 0, :]
            # beam or top_k_top_p
            if decoder_outputs['matched_eos_pos'].ndim == 2:
                matched_eos_pos = decoder_outputs['matched_eos_pos'][0]
            else:
                matched_eos_pos = decoder_outputs['matched_eos_pos']
            for index, single_tensor in enumerate(predicted_ids):
                eos_index = matched_eos_pos[index]
                predicted_ids_sliced.append(single_tensor[:eos_index].numpy().tolist())

            predicted_summaries_text = self.tokenizer.batch_decode(predicted_ids_sliced, skip_special_tokens=True)
            predicted_summaries.extend(predicted_summaries_text)

            original_decoded = self.tokenizer.batch_decode(batch_labels['labels'].numpy())
            print("original_decoded", original_decoded)
            original_summaries.extend(original_decoded)

        assert len(original_summaries) == len(predicted_summaries)
        df = pd.DataFrame()
        df['original_summaries'] = original_summaries
        df['predicted_summaries'] = predicted_summaries

        for i in range(len(original_summaries)):
            score = scorer.score(original_summaries[i], predicted_summaries[i])
            aggregator.add_scores(score)

        result = {}
        result['rouge2_f1score_mid'] = aggregator.aggregate()['rouge2'].mid.fmeasure
        result['rouge1_f1score_mid'] = aggregator.aggregate()['rouge1'].mid.fmeasure
        result['rougel_f1score_mid'] = aggregator.aggregate()['rougeLsum'].mid.fmeasure

        global_step = traininer_kwargs['global_step']
        wandb = traininer_kwargs['wandb']
        if wandb:
            wandb.log(result, step=global_step)

        return result
