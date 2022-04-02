# coding=utf-8
# Copyright 2021 TF-Transformers Authors and The TensorFlow Authors.
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
"""The main wrapper around SentenceTransformers"""
from typing import Optional, Union

from tf_transformers.models.sentence_transformers.distilbert_sentence_model import (
    SentenceBertModel,
)
from tf_transformers.models.sentence_transformers.distilroberta_model import (
    SentenceDistilRobertaModel,
)
from tf_transformers.models.sentence_transformers.minilm_model import (
    SentenceMiniLMModel,
)
from tf_transformers.models.sentence_transformers.t5_sentence_model import (
    SentenceT5Model,
)

sentence_t5_models = [
    'sentence-transformers/gtr-t5-base',
    'sentence-transformers/gtr-t5-large',
    'sentence-transformers/gtr-t5-xl',
    'sentence-transformers/sentence-t5-base',
    'sentence-transformers/sentence-t5-large',
    'sentence-transformers/sentence-t5-xl',
]

sentence_distilbert_model = [
    'sentence-transformers/msmarco-distilbert-dot-v5',
    'sentence-transformers/stsb-distilbert-base',
    'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
    'sentence-transformers/msmarco-distilbert-base-v4',
    'sentence-transformers/msmarco-distilbert-cos-v5',
    'sentence-transformers/msmarco-distilbert-base-v2',
    'sentence-transformers/distilbert-base-nli-mean-tokens',
    'sentence-transformers/multi-qa-distilbert-cos-v1',
]

sentence_minilm_models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-MiniLM-L6-v1',
    'sentence-transformers/all-MiniLM-L12-v1',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/multi-qa-MiniLM-L6-dot-v1',
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'sentence-transformers/paraphrase-MiniLM-L3-v2',
    'sentence-transformers/paraphrase-MiniLM-L6-v2',
    'sentence-transformers/paraphrase-MiniLM-L12-v2',
    'sentence-transformers/msmarco-MiniLM-L6-cos-v5',
    'sentence-transformers/msmarco-MiniLM-L12-cos-v5',
]

sentence_distilroberta_models = ['sentence-transformers/all-distilroberta-v1']


class SentenceTransformer:
    """Sentence Transformer"""

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        return_all_outputs=False,
        cache_dir: Union[str, None] = None,
        model_checkpoint_dir: Optional[str] = None,
        convert_from_hf: bool = True,
        return_layer: bool = False,
        return_config: bool = False,
        convert_fn_type: Optional[str] = "pt",
        save_checkpoint_cache: bool = True,
        load_from_cache: bool = True,
        skip_hub=False,
        **kwargs,
    ):
        # Raise value error if model_name is unknown
        if model_name in sentence_t5_models:

            model = SentenceT5Model.from_pretrained(
                model_name=model_name,
                return_all_outputs=return_all_outputs,
                cache_dir=cache_dir,
                model_checkpoint_dir=model_checkpoint_dir,
                convert_from_hf=convert_from_hf,
                return_layer=return_layer,
                return_config=return_config,
                convert_fn_type=convert_fn_type,
                save_checkpoint_cache=save_checkpoint_cache,
                load_from_cache=load_from_cache,
                skip_hub=skip_hub,
                **kwargs,
            )
            return model

        if model_name in sentence_distilbert_model:

            model = SentenceBertModel.from_pretrained(
                model_name=model_name,
                return_all_outputs=return_all_outputs,
                cache_dir=cache_dir,
                model_checkpoint_dir=model_checkpoint_dir,
                convert_from_hf=convert_from_hf,
                return_layer=return_layer,
                return_config=return_config,
                convert_fn_type=convert_fn_type,
                save_checkpoint_cache=save_checkpoint_cache,
                load_from_cache=load_from_cache,
                skip_hub=skip_hub,
                **kwargs,
            )
            return model

        if model_name in sentence_minilm_models:

            model = SentenceMiniLMModel.from_pretrained(
                model_name=model_name,
                return_all_outputs=return_all_outputs,
                cache_dir=cache_dir,
                model_checkpoint_dir=model_checkpoint_dir,
                convert_from_hf=convert_from_hf,
                return_layer=return_layer,
                return_config=return_config,
                convert_fn_type=convert_fn_type,
                save_checkpoint_cache=save_checkpoint_cache,
                load_from_cache=load_from_cache,
                skip_hub=skip_hub,
                **kwargs,
            )
            return model

        if model_name in sentence_distilroberta_models:

            model = SentenceDistilRobertaModel.from_pretrained(
                model_name=model_name,
                return_all_outputs=return_all_outputs,
                cache_dir=cache_dir,
                model_checkpoint_dir=model_checkpoint_dir,
                convert_from_hf=convert_from_hf,
                return_layer=return_layer,
                return_config=return_config,
                convert_fn_type=convert_fn_type,
                save_checkpoint_cache=save_checkpoint_cache,
                load_from_cache=load_from_cache,
                skip_hub=skip_hub,
                **kwargs,
            )
            return model

        return ValueError("Model name `{}` not found in tf-transformers/Sentence-transformers".format(model_name))
