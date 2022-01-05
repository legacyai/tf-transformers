
#### Start TPU


mkdir Libraries
mkdir Projects
mkdir Datasets

alias python=python3
alias pip=pip3

pip3 install datasets
pip3 install transformers
pip3 install sentencepiece
pip3 install tensorflow-hub # For tf-text
pip3 install jupyter

alias ipython=~/.local/bin/ipython

#### Install TF-text by building with bazel

cd Libraries

git clone https://github.com/s4sarath/text.git
cd text
./oss_scripts/run_build.sh
cd text/tensorflow_text/python/ops
cp ~/Libraries/text/bazel-bin/tensorflow_text/python/ops/*.so .

cd text/tensorflow_text/python/metrics
cp ~/Libraries/text/bazel-bin/tensorflow_text/python/metrics/_text_similarity_metric_ops.so .

export PYTHONPATH=$PYTHONPATH:/home/sidhu/Libraries/text/

#### Install tf-transformers

cd Libraries
git clone -b modification https://github.com/legacyai/tf-transformers.git
export PYTHONPATH=$PYTHONPATH:/home/sidhu/Libraries/tf-transformers/src/



##### Download TFRecord and vocab from gsbucket
cd Datasets
gsutil -m cp -r "gs://tft_free/PRETRAIN_DATA/" .


nohup python3 /path/to/test.py > output.log &
