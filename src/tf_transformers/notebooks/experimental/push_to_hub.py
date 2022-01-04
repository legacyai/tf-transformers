from tf_transformers.models import ViTModel
model_name = 'vit-large-patch32-384'
model = ViTModel.from_pretrained(model_name)

model = ViTModel.from_pretrained(model_name, classification_labels=1000)


import os
import time
import subprocess
from distutils.dir_util import copy_tree
from absl import logging
logging.set_verbosity("INFO")

models_list = [
               'vit-base-patch16-224',
               'vit-base-patch32-384',
               'vit-base-patch32-224-in21k',
               'vit-large-patch16-224',
               'vit-large-patch32-384'
               ]
cwd = os.getcwd()
MODEL_DIR = '/home/sarathrnair/MODELS/'

for model_name in models_list:

    subprocess.run(['huggingface-cli', 'repo',
                    'create', '{}'.format(model_name),
                    '--yes',
                    '--organization', 'tftransformers'])

    subprocess.run(["git", "clone", "https://huggingface.co/tftransformers/{}".format(model_name)])
    new_working_dir = os.path.join(cwd, model_name)
    os.chdir("{}".format(new_working_dir))
    cached_model_dir = os.path.join(MODEL_DIR, "tf_transformers_cache/{}/".format(model_name))
    copy_tree(cached_model_dir, new_working_dir)

    subprocess.run(["git-lfs", "track", "*"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "New Model"])
    subprocess.run(["git", "push"])

    os.chdir("{}".format(cwd))
    time.sleep(2)
    logging.info("Completed {}".format(model_name))
    print("------------------------------------------------------------------")