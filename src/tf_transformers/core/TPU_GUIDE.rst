
gcloud config set compute/zone us-central1-f


export PROJECT_ID=tft-tpu

gcloud config set project ${PROJECT_ID}

gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID

# Service identity created: service-921849362698@cloud-tpu.iam.gserviceaccount.com
