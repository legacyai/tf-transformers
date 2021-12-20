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
"""Evaluation script for t5"""
import os
import tempfile

import hydra
import pandas as pd
import tensorflow as tf
import tqdm
from dataset_loader import load_dataset_eval
from model import get_model_inference, get_tokenizer
from omegaconf import DictConfig
from rouge_score import rouge_scorer, scoring

from tf_transformers.text import TextDecoder


def predict_and_evaluate(decoder, eval_dataset, tokenizer, decoder_seq_len, eos_id, mode):
    """Predict and evaluate"""
    predicted_summaries = []
    original_summaries = []
    for (batch_inputs, batch_labels) in tqdm.tqdm(eval_dataset):

        del batch_inputs[
            'decoder_input_ids'
        ]  # We do not need to pass decoder_input_ids , as we provide while initiating
        # TextDecoder

        if mode == 'greedy':
            decoder_outputs = decoder.decode(
                batch_inputs, max_iterations=decoder_seq_len + 1, mode='greedy', eos_id=eos_id
            )

        else:
            decoder_outputs = decoder.decode(
                batch_inputs, max_iterations=decoder_seq_len + 1, mode='beam', num_beams=3, alpha=0.8, eos_id=eos_id
            )

        predicted_batch_summaries = tokenizer._tokenizer.detokenize(decoder_outputs['predicted_ids'][:, 0, :].numpy())
        predicted_summaries.extend(predicted_batch_summaries.numpy().tolist())

        original_batch_summaries = tokenizer._tokenizer.detokenize(batch_labels['labels'])
        original_summaries.extend(original_batch_summaries.numpy().tolist())

    predicted_summaries = [entry.decode() for entry in predicted_summaries]
    original_summaries = [text.decode() for text in original_summaries]

    df = pd.DataFrame()
    df['original_summaries'] = original_summaries
    df['predicted_summaries'] = predicted_summaries
    df.to_csv("prediction_summaries.csv", index=False)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for i in range(len(original_summaries)):
        score = scorer.score(original_summaries[i], predicted_summaries[i])
        aggregator.add_scores(score)

    result = {}
    result['rouge2_f1score_mid'] = aggregator.aggregate()['rouge2'].mid.fmeasure
    result['rouge1_f1score_mid'] = aggregator.aggregate()['rouge1'].mid.fmeasure
    result['rougel_f1score_mid'] = aggregator.aggregate()['rougeLsum'].mid.fmeasure
    return result


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    print("Config", cfg)

    model_name = cfg.model.model_name
    num_splits = cfg.task.num_splits
    use_gru_layer = cfg.task.use_gru_layer
    projection_dimension = cfg.task.projection_dimension

    max_seq_len = cfg.task.max_seq_len
    decoder_seq_len = cfg.task.decoder_seq_len

    eval_batch_size = cfg.eval.eval_batch_size
    model_checkpoint_dir = cfg.eval.model_checkpoint_dir
    model_checkpoint_path = cfg.eval.model_checkpoint_path
    take_sample = cfg.eval.take_sample
    mode = cfg.eval.mode

    temp_dir = tempfile.TemporaryDirectory().name

    if model_checkpoint_path and model_checkpoint_dir:
        raise ValueError("Do not provide both `model_checkpoint_path` and `model_checkpoint_dir`.")

    if max_seq_len % num_splits != 0:
        raise ValueError("`num_splits` should be divisble by `max_seq_len`")

    tokenizer_layer = get_tokenizer(model_name, max_seq_len)

    # Get Inference Model
    model_inference = get_model_inference(model_name, num_splits, use_gru_layer, projection_dimension)
    eval_dataset, _ = load_dataset_eval(tokenizer_layer, max_seq_len, decoder_seq_len, eval_batch_size)
    if take_sample:
        eval_dataset = eval_dataset.take(20)  # We take only 20 after batching for callbacks

    if model_checkpoint_dir:
        all_results = []
        print("Model model_checkpoint_dir", model_checkpoint_dir)
        number_of_checkpoints = int(
            tf.train.latest_checkpoint(model_checkpoint_dir).split("/")[-1].replace("ckpt-", "")
        )
        number_of_checkpoints += 1
        for checkpoint_number in range(1, number_of_checkpoints):
            ckpt_path = os.path.join(model_checkpoint_dir, "ckpt-{}".format(checkpoint_number))
            model_inference.load_checkpoint(checkpoint_path=ckpt_path)

            # Save as serialized module
            model_inference.save_transformers_serialized(temp_dir, overwrite=True)

            model_pb = tf.saved_model.load(temp_dir)
            decoder = TextDecoder(model_pb)

            result = predict_and_evaluate(
                decoder, eval_dataset, tokenizer_layer, decoder_seq_len, tokenizer_layer.eos_token_id, mode
            )
            all_results.append(result)
            print("ckpt_path: ", ckpt_path)
            print(result)
            print()

        print()
        print("Final Result")
        print(all_results)

    elif model_checkpoint_path:
        model_inference.load_checkpoint(checkpoint_path=model_checkpoint_path)

        # Save as serialized module
        model_inference.save_transformers_serialized(temp_dir, overwrite=True)

        model_pb = tf.saved_model.load(temp_dir)
        decoder = TextDecoder(model_pb)

        result = predict_and_evaluate(
            decoder, eval_dataset, tokenizer_layer, decoder_seq_len, tokenizer_layer.eos_token_id, mode
        )
        print("ckpt_path: ", model_checkpoint_path)
        print(result)


if __name__ == "__main__":
    run()
