..
    Copyright 2020 The HuggingFace Team and TFT Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

BigBird Roberta Tokenizer
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page includes information about how to use AlbertTokenizer with tensorflow-text.
This tokenizer works in sync with :class:`~tf.data.Dataset` and so is useful for on the fly tokenization.
This tokenizer is more of like `GPT2` vocab extension in sentencepiece with 100 extra reserved IDS.

.. code-block::

        >>> from tf_transformers.models import  BigBirdTokenizerTFText
        >>> tokenizer = BigBirdTokenizerTFText.from_pretrained("google/bigbird-roberta-large")
        >>> text = ['The following statements are true about sentences in English:',
                    '',
                    'A new sentence begins with a capital letter.']
        >>> inputs = {'text': text}
        >>> outputs = tokenizer(inputs) # Ragged Tensor Output

        # Dynamic Padding
        >>> tokenizer = BigBirdTokenizerTFText.from_pretrained("google/bigbird-roberta-large",
        dynamic_padding=True)
        >>> text = ['The following statements are true about sentences in English:',
                    '',
                    'A new sentence begins with a capital letter.']
        >>> inputs = {'text': text}
        >>> outputs = tokenizer(inputs) # Dict of tf.Tensor

        # Static Padding
        >>> tokenizer = BigBirdTokenizerTFText.from_pretrained("google/bigbird-roberta-large",
        pack_model_inputs=True)
        >>> text = ['The following statements are true about sentences in English:',
                    '',
                    'A new sentence begins with a capital letter.']
        >>> inputs = {'text': text}
        >>> outputs = tokenizer(inputs) # Dict of tf.Tensor

        # To Add Special Tokens
        >>> tokenizer = BigBirdTokenizerTFText.from_pretrained("google/bigbird-roberta-large",
        add_special_tokens=True)

BigBirdRobertaTokenizerTFText
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.BigBirdRobertaTokenizerTFText
    :members:

BigBirdRobertaTokenizerLayer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.BigBirdRobertaTokenizerLayer
    :members:
