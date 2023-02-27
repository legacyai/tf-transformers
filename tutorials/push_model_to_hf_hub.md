---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3

```

```{code-cell} ipython3
import subprocess
import os
from distutils.dir_util import copy_tree
```

### How to push  a model to hub  .

* Make sure you have logged to ```huggingface-cli login``` using your token.

```{code-cell} ipython3

```

```{code-cell} ipython3
model_name = 'byt5-small'
```

#### 1. Create model name directory under organization name

```{code-cell} ipython3
subprocess.run(['huggingface-cli', 'repo',
                'create', '{}'.format(model_name),
                '--yes',
                '--organization', 'tftransformers'])
```

```{code-cell} ipython3

```

#### 2. Now clone that above created repo/folder to our local cwd

```{code-cell} ipython3
subprocess.run(["git", "clone", "https://huggingface.co/tftransformers/{}".format(model_name)])
```

```{code-cell} ipython3

```

#### 3. Now move your model directory , to current working directory under ```model_name``` directory

```{code-cell} ipython3
cwd = os.getcwd() # Getc current working dir
new_working_dir = os.path.join(cwd, model_name) # This is cloned from hf hub under organization
os.chdir("{}".format(new_working_dir)) # Switch to new working dir

# Cached model directory keep changing as per other machine
cached_model_dir = '/var/folders/vq/4fxns8l55gq8_msgygbyb51h0000gn/T/tf_transformers_cache/{}/'.format(model_name)

# Copy cached model directory , to new working directory
copy_tree(cached_model_dir, new_working_dir)
```

```{code-cell} ipython3

```

#### 4. Now time to push these model to hub

```{code-cell} ipython3
subprocess.run(["git-lfs", "track", "*"])
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "Pushing new model {}".format(model_name)]) # Commit message
subprocess.run(["git", "push"])

# Change back to original cwd 
os.chdir("{}".format(cwd))
```

```{code-cell} ipython3

```
