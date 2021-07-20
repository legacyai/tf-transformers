
URL: https://cloud.google.com/tpu/docs/tensorflow-quickstart-tpu-vm


1. Install Google Cloud SDK

URL: https://cloud.google.com/sdk/docs/install

a. curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-347.0.0-linux-x86_64.tar.gz

b. tar -xvf google-cloud-sdk-347.0.0-linux-x86_64.tar.gz

c. ./google-cloud-sdk/install.sh

d. ./google-cloud-sdk/bin/gcloud init

------------------------------------------------------------------------------------------------------------------


a. gcloud config set project tft-tpu

b. gcloud services enable tpu.googleapis.com

c. gcloud config set account your-email-account

d. gcloud config set project tft-tpu

# tpu-name = tft-free-tpu
e. gcloud alpha compute tpus tpu-vm create tft-free-tpu \
  --zone=europe-west4-a \
  --accelerator-type=v3-8 \
  --version=v2-alpha


f . gcloud alpha compute tpus tpu-vm ssh tft-free-tpu --zone europe-west4-a --project tft-tpu

# Check status

gcloud compute tpus list --zone=europe-west4-a

g. gcloud alpha compute tpus tpu-vm delete tft-free-tpu \
  --zone=europe-west4-a


sudo python3 -m pip3 install google-cloud-storage

!gsutil -m cp -r logs/ gs://tft_free/logs_new


# Learning rate

https://aws.amazon.com/blogs/machine-learning/the-importance-of-hyperparameter-tuning-for-scaling-deep-learning-training-to-multiple-gpus/#:~:text=The%20importance%20of%20hyperparameter%20tuning%20for%20scaling%20deep%20learning%20training%20to%20multiple%20GPUs,-by%20Sina%20Afrooze&text=Parallel%20processing%20with%20multiple%20GPUs,scaling%20training%20of%20deep%20models.&text=To%20ensure%20that%20you%20fully,linearly%20with%20each%20additional%20GPU.

https://www.wwt.com/api/attachments/5eac259f31d66a008d00abe5/file

https://databricks.com/blog/2019/08/15/how-not-to-scale-deep-learning-in-6-easy-steps.html


alias python=python3
alias pip=pip3

------------------------------------------------------------------------------------
 TF Text on TPU

 "@local_config_tf//:libtensorflow_framework" -> "@org_tensorflow//tensorflow/core:framework"
"@local_config_tf//:tf_header_lib" -> "@org_tensorflow//tensorflow/core:lib"

git clone https://github.com/s4sarath/text.git

/home/sidhu/text/bazel-bin



gsutil -m cp -R "Your local directory" gs://Your bucket name


gsutil -u tft-tpu -m cp 'gs://allennlp-tensorflow-datasets/c4/en/3.0.1/*' local_datasets_dir/c4/en/3.0.1/






gcloud alpha compute tpus tpu-vm create tft-free-tpu-malayalam \
  --zone=europe-west4-a \
  --accelerator-type=v3-8 \
  --version=v2-alpha

  gcloud alpha compute tpus tpu-vm create tft-free-tpu-malayalam \
  --zone=us-central1-f  \
  --accelerator-type=v2-8 \
  --version=v2-alpha

  gcloud alpha compute tpus tpu-vm ssh tft-free-tpu-malayalam --zone us-central1-f --project tft-tpu


  gcloud alpha compute tpus tpu-vm create tft-free-tpu2 \
  --zone=europe-west4-a \
  --accelerator-type=v3-8 \
  --version=v2-alpha



  gcloud alpha compute tpus tpu-vm ssh tft-free-tpu2 --zone europe-west4-a --project tft-tpu

  python3 1_data_to_text.py data.name=bookcorpus data.version= data.output_text_file=/home/Sidhu/datasets/bookcorpus.txt
