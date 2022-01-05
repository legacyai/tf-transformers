..
    Copyright 2020 The HuggingFace Team and TFT Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

MT5
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mT5 model was presented in `mT5: A massively multilingual pre-trained text-to-text transformer
<https://arxiv.org/abs/2010.11934>`_ by Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya
Siddhant, Aditya Barua, Colin Raffel.

The abstract from the paper is the following:

*The recent "Text-to-Text Transfer Transformer" (T5) leveraged a unified text-to-text format and scale to attain
state-of-the-art results on a wide variety of English-language NLP tasks. In this paper, we introduce mT5, a
multilingual variant of T5 that was pre-trained on a new Common Crawl-based dataset covering 101 languages. We describe
the design and modified training of mT5 and demonstrate its state-of-the-art performance on many multilingual
benchmarks. All of the code and model checkpoints*

Note: mT5 was only pre-trained on `mC4 <https://huggingface.co/datasets/mc4>`__ excluding any supervised training.
Therefore, this model has to be fine-tuned before it is useable on a downstream task, unlike the original T5 model.
Since mT5 was pre-trained unsupervisedly, there's no real advantage to using a task prefix during single-task
fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.

Google has released the following variants:

- `google/mt5-small <https://huggingface.co/google/mt5-small>`__

- `google/mt5-base <https://huggingface.co/google/mt5-base>`__

- `google/mt5-large <https://huggingface.co/google/mt5-large>`__

- `google/mt5-xl <https://huggingface.co/google/mt5-xl>`__

- `google/mt5-xxl <https://huggingface.co/google/mt5-xxl>`__.


`Paper👆 <https://arxiv.org/abs/2010.11934>`__
`Official Code👆 <https://github.com/google-research/multilingual-t5>`__


MT5Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.mt5.MT5Config
    :members:


MT5Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.MT5Model
    :members:


MT5Encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.MT5Encoder
    :members:
