
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
