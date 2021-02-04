<!---
Copyright 2021 The LegacyAI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <br>
    <img src="src/logo2.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

<h3 align="center">
<p>tf-transformers: faster and easier state-of-the-art NLP in TensorFlow 2.0
</h3>

tf-transformers is designed to harness the full power of Tensorflow 2, to make it much faster and simpler comparing to existing Tensorflow based NLP architectures. On an average, there is ```80 % ``` improvement over current exsting Tensorflow based libraries, on text generation. You can find more details in the Benchmarks section.

All / Most NLP downstream tasks can be integrated into Tranformer based models with much ease. All the models can be trained using ```model.fit```, which supports GPU, multi-GPU, TPU.

## Unique Features 
- Faster text generation using Tensorflow2. Faster than PyTorch in most experiments (V100 GPU). ```80%``` faster compared to existing TF based libararies (relative difference) Refer [benchmark code](tests/notebooks/benchmarks/).
- Complete TFlite support for BERT, RoBERTA, T5, Albert, mt5 for all down stream tasks except text-generation
- Faster sentence-piece alignment (no more LCS overhead)
- Variable batch text generation for Encoder only models like GPT2
- No more hassle of writing long codes for TFRecords. minimal and simple. 
- Off the shelf support for auto-batching ```tf.data.dataset``` or ```tf.ragged```tensors
- Pass dictionary outputs directly to loss functions inside ```tf.keras.Model.fit``` using ```model.compile2``` . Refer [examples]() or [blog]()
- Multiple mask modes like ```causal```, ```user-defined```, ```prefix``` by changing one argument . Refer [examples]() or [blog]()


## Production Ready Tutorials


Here are a few examples:
- [Masked word completion with BERT](https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Name Entity Recognition with Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Text generation with GPT-2](https://huggingface.co/gpt2?text=A+long+time+ago%2C+)
- [Natural Langugage Inference with RoBERTa](https://huggingface.co/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Summarization with BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Question answering with DistilBERT](https://huggingface.co/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Translation with T5](https://huggingface.co/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)


## Quick tour

To immediately use a model on a given text, we provide the `pipeline` API. Pipelines group together a pretrained model with the preprocessing that was used during that model training. Here is how to quickly use a pipeline to classify positive versus negative texts

```python
>>> from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to include pipeline into the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9978193640708923}]
```

The second line of code downloads and caches the pretrained model used by the pipeline, the third line evaluates it on the given text. Here the answer is "positive" with a confidence of 99.8%.

This is another example of pipeline used for that can extract question answers from some context:

``` python
>>> from transformers import pipeline

# Allocate a pipeline for question-answering
>>> question_answerer = pipeline('question-answering')
>>> question_answerer({
...     'question': 'What is the name of the repository ?',
...     'context': 'Pipeline have been included in the huggingface/transformers repository'
... })
{'score': 0.5135612454720828, 'start': 35, 'end': 59, 'answer': 'huggingface/transformers'}

```

On top of the answer, the pretrained model used here returned its confidence score, along with the start position and its end position in the tokenized sentence. You can learn more about the tasks supported by the `pipeline` API in [this tutorial](https://huggingface.co/transformers/task_summary.html).

To download and use any of the pretrained models on your given task, you just need to use those three lines of codes (PyTorch version):
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = AutoModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```
or for TensorFlow:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

The tokenizer is responsible for all the preprocessing the pretrained model expects, and can be called directly on one (or list) of texts (as we can see on the fourth line of both code examples). It will output a dictionary you can directly pass to your model (which is done on the fifth line).

The model itself is a regular [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) or a [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (depending on your backend) which you can use normally. For instance, [this tutorial](https://huggingface.co/transformers/training.html) explains how to integrate such a model in classic PyTorch or TensorFlow training loop, or how to use our `Trainer` API to quickly fine-tune the on a new dataset.

## Why should I use tf-transformers?

1. Use state-of-the-art models in Production, with less than 10 lines of code.
    - High performance models, better tha all official Tensorflow based models 
    - Very simple classes for all downstream tasks
    - Complete TFlite support for all tasks except text-generation

2. Make industry based experience to avaliable to students and community with clear tutorials

3. Train any model on GPU, multi-GPU, TPU with amazing ```tf.keras.Model.fit```
    - Train state-of-the-art models in few lines of code.
    - All models are completely serializable.


4. Customize any models or pipelines with minimal or no code change.


## Installation

### With pip

This repository is tested on Python 3.7+, and Tensorflow 2.4.0

Recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html). 

Assuming Tensorflow 2.0 is installed

```bash
pip install tf-transformers
```


## Supported Models architectures




tf-transformers currently provides the following architectures .
1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
2. **[BERT](https://huggingface.co/transformers/model_doc/bert.html)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
3. **[BERT For Sequence Generation](https://huggingface.co/transformers/model_doc/bertgeneration.html)** (from Google) released with the paper [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
4. **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)** (from Google Research/Stanford University) released with the paper [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) by Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
5. **[GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)** (from OpenAI) released with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
6. **[MT5](https://huggingface.co/transformers/model_doc/mt5.html)** (from Google AI) released with the paper [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934) by Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.
7. **[RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)** (from Facebook), released together with the paper a [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
ultilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation) and a German version of DistilBERT.
**[T5](https://huggingface.co/transformers/model_doc/t5.html)** (from Google AI) released with the paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.

These implementations have been tested on several datasets (see the example scripts) and should match the performances of the original implementations. You can find more details on the performances in the Examples section of the [documentation](https://huggingface.co/transformers/examples.html).


## Learn more

| Section | Description |
|-|-|
| [Documentation](https://huggingface.co/transformers/) | Full API documentation and tutorials |
| [Task summary](https://huggingface.co/transformers/task_summary.html) | Tasks supported by 🤗 Transformers |
| [Preprocessing tutorial](https://huggingface.co/transformers/preprocessing.html) | Using the `Tokenizer` class to prepare data for the models |
| [Training and fine-tuning](https://huggingface.co/transformers/training.html) | Using the models provided by 🤗 Transformers in a PyTorch/TensorFlow training loop and the `Trainer` API |
| [Quick tour: Fine-tuning/usage scripts](https://github.com/huggingface/transformers/tree/master/examples) | Example scripts for fine-tuning models on a wide range of tasks |
| [Model sharing and uploading](https://huggingface.co/transformers/model_sharing.html) | Upload and share your fine-tuned models with the community |
| [Migration](https://huggingface.co/transformers/migration.html) | Migrate to 🤗 Transformers from `pytorch-transformers` or `pytorch-pretrained-bert` |

## Citation

Nothing here yet :-) . 
```
