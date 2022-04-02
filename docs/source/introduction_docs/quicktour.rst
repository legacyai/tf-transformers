..
    Copyright 2020 TFT and The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Quick tour
=======================================================================================================================

Let's have a quick look at the tf-transformers library features. The library downloads pretrained models for Natural
Language Understanding (NLU) tasks, such as analyzing the sentiment of a text, and Natural Language Generation (NLG),
such as completing a prompt with new text or translating in another language.


.. note::

    tf-transformers is written in such a way that it does **in-place** conversion of HuggingFace models on the fly,
    to tf-transformers checkpoints. If **Tensorflow** models are not present, it will do the conversion on
    **PyTorch** models.


tf-transformers provides the following tasks out of the box:

- Text Classification
- Text generation , Summarization, NLG: provide a prompt and the model will generate what follows.
- Name entity recognition (NER): in an input sentence, label each word with the entity it represents (person, place,
  etc.)
- Question answering: provide the model with some context and a question, extract the answer from the context.
- Feature extraction: return a tensor representation of the text.

We can easily load a model in 2 lines of code. For training:

.. code-block::

    >>> from tf_transformers.models import GPT2Model
    >>> model = GPT2Model.from_pretrained('gpt2')

All model checkpoints can be saved using :obj:`model.save_checkpoint`. For saving checkpoints:

.. code-block::

    >>> from tf_transformers.models import GPT2Model
    >>> model = GPT2Model.from_pretrained('gpt2')
    >>> model.save_checkpoint("/tmp/gpt2_model/")

All model checkpoints can be loading using :obj:`model.load_checkpoint`. For loading checkpoints:

.. code-block::

    >>> from tf_transformers.models import GPT2Model
    >>> model = GPT2Model.from_pretrained('gpt2')
    >>> model.load_checkpoint("/tmp/gpt2_model/")

When typing this command for the first time, a pretrained model will be downloaded and cached. For training, this is
how should be used.

For inference, it is very important to add :obj:`use_auto_regressive=True`. This is required for all the models,
if you are planning to do text generation or auto regressive tasks:
.. code-block::

    >>> from tf_transformers.models import GPT2Model
    >>> model = GPT2Model.from_pretrained('gpt2', use_auto_regressive=True)

All models in tf-transformers are designed to be completely serializable to .pb format. This is where the performance
benefits of tf-transformers models comes into picture. It is recommened to use :obj:(`.pb serializable`) models for
best performance. To serialize:

.. code-block::

    >>> from tf_transformers.models import GPT2Model
    >>> model = GPT2Model.from_pretrained('gpt2')
    >>> model.save_transformers_serialized("/tmp/gpt2_serialized/")

To load a serialized models for inference in prodcution:

.. code-block::

    >>> import tensorflow as tf
    >>> loaded = tf.saved_model.load("/tmp/gpt2_serialized/")
    >>> model  = loaded.signatures['serving_default']
