<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

# Installation

tf-ransformers is tested on Python 3.8+, and TensorFlow 2.7.1+.

You should install tf-ransformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're
unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Create a virtual environment with the version of Python you're going
to use and activate it.

Now, if you want to use tf-transformers, you can install it with pip. If you'd like to play with the examples, you
must install it from source.

## Installation with pip

First you need to install  TensorFlow 2.0 .
Some in-place conversions require PyTorch, which depends on the model.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available),
[PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) and/or
regarding the specific install command for your platform.

When TensorFlow 2.7.1 (CPU or GPU) has been installed, tf-ransformers can be installed using pip as follows:

```bash
pip install tf-transformers
```


## Editable install

If you want to constantly use the bleeding edge `master` version of the source code, or if you want to contribute to the library and need to test the changes in the code you're making, you will need an editable install. This is done by cloning the repository and installing with the following commands:

First you need to install poetry
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

``` bash
git clone https://github.com/legacyai/tf-transformers.git
cd tf-transformers
poetry install
```


## Caching models

This library provides pretrained models that will be downloaded and cached locally. Unless you specify a location with
`cache_dir=...` when you use methods like `from_pretrained`, these models will automatically be downloaded in the
``tf_transformers_cache``.

  * default: ``/tmp/tf_transformers_cache/`` . ``/tmp`` in Ubuntu, might depends on the temp_dir in the machine.


## Do you want to run a tf-ransformer model on a mobile device?

All models in tf-transformers completely support TFlite capbaility. Please checkout the tutorials for in-depth details.
