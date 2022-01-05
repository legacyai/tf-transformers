..
    Copyright 2020 The HuggingFace Team and TFT Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

ALBERT Tokenizer
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page includes information about how to use AlbertTokenizer with tensorflow-text.
This tokenizer works in sync with :class:`~tf.data.Dataset` and so is useful for on the fly tokenization.

.. code-block::

      from tf_transformers.models import  AlbertTokenizerTFText
      tokenizer = AlbertTokenizerTFText.from_pretrained("albert-base-v2")
      text = ['The following statements are true about sentences in English:',
                  '',
                  'A new sentence begins with a capital letter.']
      # All tokenizer expects a dictionary
      inputs = {'text': text}
      outputs = tokenizer(inputs) # Ragged Tensor Output

      # Dynamic Padding
      tokenizer = AlbertTokenizerTFText.from_pretrained("albert-base-v2", dynamic_padding=True)
      text = ['The following statements are true about sentences in English:',
                  '',
                  'A new sentence begins with a capital letter.']
      inputs = {'text': text}
      outputs = tokenizer(inputs) # Dict of tf.Tensor

      # Static Padding
      tokenizer = AlbertTokenizerTFText.from_pretrained("albert-base-v2", pack_model_inputs=True)
      text = ['The following statements are true about sentences in English:',
                  '',
                  'A new sentence begins with a capital letter.']
      inputs = {'text': text}
      outputs = tokenizer(inputs) # Dict of tf.Tensor

      # To Add Special Tokens
      tokenizer = AlbertTokenizerTFText.from_pretrained("albert-base-v2", add_special_tokens=True)


AlbertTokenizerTFText
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.AlbertTokenizerTFText
    :members:

AlbertTokenizerLayer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.AlbertTokenizerLayer
    :members:
