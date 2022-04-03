..
    Copyright 2020 The HuggingFace Team and TFT Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Sentence Transformer
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the tensorflow implementation of [Sentence Transformer](https://www.sbert.net/).
As of now ```tf-transformers``` support following models.
.. code-block::

    'sentence-transformers/gtr-t5-base',
    'sentence-transformers/gtr-t5-large',
    'sentence-transformers/gtr-t5-xl',
    'sentence-transformers/sentence-t5-base',
    'sentence-transformers/sentence-t5-large',
    'sentence-transformers/sentence-t5-xl',
    'sentence-transformers/msmarco-distilbert-dot-v5',
    'sentence-transformers/stsb-distilbert-base',
    'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
    'sentence-transformers/msmarco-distilbert-base-v4',
    'sentence-transformers/msmarco-distilbert-cos-v5',
    'sentence-transformers/msmarco-distilbert-base-v2',
    'sentence-transformers/distilbert-base-nli-mean-tokens',
    'sentence-transformers/multi-qa-distilbert-cos-v1',
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
    'sentence-transformers/all-distilroberta-v1'

.. code-block::


Usage:

.. code-block::

    >>> from tf_transformers.models import SentenceTransformer
    >>> model = SentenceTransformer.from_pretrained('sentence-transformers/msmarco-MiniLM-L6-cos-v5')

All model checkpoints can be saved using :obj:`model.save_checkpoint`. For saving checkpoints:

.. code-block::



SentenceTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.SentenceTransformer
    :members:
