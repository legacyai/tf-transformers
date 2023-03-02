Tensorflow Transformers (tf-transformers)
=======================================================================================================================

State-of-the-art Faster Natural Language Processing in TensorFlow 2.0.

tf-transformers  provides general-purpose
architectures (BERT, GPT-2, RoBERTa, T5, Seq2Seq...) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages in TensorFlow 2.0.

tf-transformers is the fastest library for Transformer based architectures, comparing to existing similar
implementations in TensorFlow 2.0. It is 80x faster comparing to famous similar libraries like HuggingFace Tensorflow
2.0 implementations. For more details about benchmarking please look `BENCHMARK` here.

This is the documentation of our repository `tf-transformers <https://github.com/legacyai/tf-transformers>`. You can
also follow our documentation <https://legacyai.github.com/tf-transformers? that teaches how to use this library, as well as the
other features of this library.


Features
-----------------------------------------------------------------------------------------------------------------------

- High performance on NLU and NLG tasks
- Low barrier to entry for educators and practitioners

State-of-the-art NLP for everyone:

- Deep learning researchers
- Hands-on practitioners
- AI/ML/NLP teachers and educators

..
    Copyright 2020 TFT team and The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Lower compute costs, smaller carbon footprint:

- Researchers can share trained models instead of always retraining
- Practitioners can reduce compute time and production costs
- 8 architectures with over 30 pretrained models, some in more than 100 languages

Choose the right framework for every part of a model's lifetime:

- Train state-of-the-art models in 3 lines of code
- Complete support for Tensorflow 2.0 models.
- Seamlessly pick the right framework for training, evaluation, production


Contents
-----------------------------------------------------------------------------------------------------------------------

The documentation is organized in five parts:

- **GET STARTED** contains a quick tour, the installation instructions and some useful information about our philosophy
  and a glossary.
- **MODELS** contains general documentation on how to use the library.
- **MODEL USAGE** contains quick examples on how to use the models.
- **ADVANCED TUTORIALS** contains more advanced guides that are more specific to training and inference in production.
- **RESEARCH** focuses on tutorials that have less to do with how to use the library but more about general research in
  transformers model, most written in fast pre-process and TPU
- **TFLITE** contains quick examples on how to use tflite models.
- **BENCHMARK** contains quick examples on how to benchmark models and the results.

The library currently contains Tensorflow implementations, pretrained model weights, usage scripts, tutorials and
conversion utilities for the following models.

Supported models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..

1. :doc:`ALBERT <model_doc/albert>` (from Google Research and the Toyota Technological Institute at Chicago) released
   with the paper `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
   <https://arxiv.org/abs/1909.11942>`__, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush
   Sharma, Radu Soricut.
2. :doc:`BART <model_doc/bart>` (from Facebook) released with the paper `BART: Denoising Sequence-to-Sequence
   Pre-training for Natural Language Generation, Translation, and Comprehension
   <https://arxiv.org/pdf/1910.13461.pdf>`__ by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman
   Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
3. :doc:`BERT <model_doc/bert>` (from Google) released with the paper `BERT: Pre-training of Deep Bidirectional
   Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ by Jacob Devlin, Ming-Wei Chang,
   Kenton Lee and Kristina Toutanova.
4. :doc:`BERT For Sequence Generation <model_doc/bertgeneration>` (from Google) released with the paper `Leveraging
   Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi
   Narayan, Aliaksei Severyn.
5.  :doc:`CLIP <model_doc/clip>` (from OpenAI) released with the paper `Learning Transferable Visual Models From
    Natural Language Supervision <https://arxiv.org/abs/2103.00020>`__ by Alec Radford, Jong Wook Kim, Chris Hallacy,
    Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
    Krueger, Ilya Sutskever.
6.  :doc:`GPT-2 <model_doc/gpt2>` (from OpenAI) released with the paper `Language Models are Unsupervised Multitask
    Learners <https://blog.openai.com/better-language-models/>`__ by Alec Radford*, Jeffrey Wu*, Rewon Child, David
    Luan, Dario Amodei** and Ilya Sutskever**.
7.  :doc:`M2M100 <model_doc/m2m_100>` (from Facebook) released with the paper `Beyond English-Centric Multilingual
    Machine Translation <https://arxiv.org/abs/2010.11125>`__ by by Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi
    Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman
    Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin.
8.  :doc:`MarianMT <model_doc/marian>` Machine translation models trained using `OPUS <http://opus.nlpl.eu/>`__ data by
    Jörg Tiedemann. The `Marian Framework <https://marian-nmt.github.io/>`__ is being developed by the Microsoft
    Translator Team.
9.  :doc:`MBart <model_doc/mbart>` (from Facebook) released with the paper `Multilingual Denoising Pre-training for
    Neural Machine Translation <https://arxiv.org/abs/2001.08210>`__ by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li,
    Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
10. :doc:`MBart-50 <model_doc/mbart>` (from Facebook) released with the paper `Multilingual Translation with Extensible
    Multilingual Pretraining and Finetuning <https://arxiv.org/abs/2008.00401>`__ by Yuqing Tang, Chau Tran, Xian Li,
    Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, Angela Fan.
11. :doc:`MT5 <model_doc/mt5>` (from Google AI) released with the paper `mT5: A massively multilingual pre-trained
    text-to-text transformer <https://arxiv.org/abs/2010.11934>`__ by Linting Xue, Noah Constant, Adam Roberts, Mihir
    Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.
12. :doc:`RoBERTa <model_doc/roberta>` (from Facebook), released together with the paper a `Robustly Optimized BERT
    Pretraining Approach <https://arxiv.org/abs/1907.11692>`__ by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar
    Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
13. :doc:`T5 <model_doc/t5>` (from Google AI) released with the paper `Exploring the Limits of Transfer Learning with a
    Unified Text-to-Text Transformer <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel and Noam Shazeer and Adam
    Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.
14. :doc:`Vision Transformer (ViT) <model_doc/vit>` (from Google AI) released with the paper `An Image is Worth 16x16
    Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`__ by Alexey Dosovitskiy,
    Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias
    Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.

.. toctree::
    :maxdepth: 2
    :caption: Get started

    introduction_docs/quicktour
    introduction_docs/installation
    introduction_docs/philosophy

.. toctree::
    :maxdepth: 2
    :caption: Models

    model_doc/albert
    model_doc/bert
    model_doc/gpt2
    model_doc/t5
    model_doc/mt5.rst
    model_doc/roberta.rst
    model_doc/vit.rst
    model_doc/clip.rst
    model_doc/sentence_transformer.rst

.. toctree::
    :maxdepth: 2
    :caption: Tutorials

    tutorials/1_read_write_tfrecords
    tutorials/2_text_classification_imdb_albert
    tutorials/3_masked_lm_tpu
    tutorials/4_image_classification_vit_multi_gpu
    tutorials/5_sentence_embedding_roberta_quora_zeroshot
    tutorials/6_prompt_engineering_clip
    tutorials/7_gpt2_question_answering_squad
    tutorials/8_code_code_java_to_csharp_t5
    tutorials/9_images_tfrecords
    tutorials/push_model_to_hub

.. toctree::
    :maxdepth: 2
    :caption: TFLite

    tflite_tutorials/albert_tflite
    tflite_tutorials/bert_tflite
    tflite_tutorials/roberta_tflite

.. toctree::
    :maxdepth: 2
    :caption: Model Usage

    model_usage/text_generation_using_gpt2
    model_usage/text_generation_using_t5
    model_usage/sentence_transformers

.. toctree::
    :maxdepth: 2
    :caption: Tokenizers

    model_doc/albert_tokenizer
    model_doc/bigbird_tokenizer
    model_doc/t5_tokenizer
    model_doc/clip_feature_extractor
    model_doc/vit_feature_extractor


.. toctree::
    :maxdepth: 2
    :caption: Research

    research/glue
    research/long_block_sequencer

.. toctree::
    :maxdepth: 2
    :caption: Benchmarks

    benchmarks/gpt2
    benchmarks/t5
    benchmarks/albert
    benchmarks/vit
    benchmarks/imagenet_clip_benchmark
