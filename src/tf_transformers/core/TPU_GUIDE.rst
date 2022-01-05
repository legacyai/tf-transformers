
gcloud config set compute/zone us-central1-f


export PROJECT_ID=stalwart-bliss-329805
gcloud config set project ${PROJECT_ID}

gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID

# Service identity created: service-921849362698@cloud-tpu.iam.gserviceaccount.com

gsutil mb -p ${PROJECT_ID} -c standard -l europe-west4 -b on gs://tft_free

###
# gcloud compute tpus execution-groups create \
#  --name=bert-tutorial \
#  --zone=europe-west4-a \
#  --tf-version=2.5.0 \
#  --machine-type=n1-standard-1 \
#  --accelerator-type=v3-8


 gcloud compute tpus execution-groups create \
 --name=bert-tutorial \
 --zone=us-central1-f \
 --tf-version=2.5.0 \
 --machine-type=n1-standard-1 \
 --accelerator-type=v2-8

 gcloud compute ssh bert-tutorial --zone=us-central1-f

 # gcloud compute ssh bert-tutorial --zone=europe-west4-a

export TPU_NAME=bert-tutorial

sudo pip3 install -r /usr/share/models/official/requirements.txt

export STORAGE_BUCKET=gs://tft_free

export PYTHONPATH="${PYTHONPATH}:/usr/share/models"
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16
export MODEL_DIR=${STORAGE_BUCKET}/bert-output
export GLUE_DIR=gs://cloud-tpu-checkpoints/bert/classification
export TASK=mnli

python3 /usr/share/models/official/nlp/bert/run_classifier.py \
  --mode='train_and_eval' \
  --input_meta_data_path=${GLUE_DIR}/${TASK}_meta_data \
  --train_data_path=${GLUE_DIR}/${TASK}_train.tf_record \
  --eval_data_path=${GLUE_DIR}/${TASK}_eval.tf_record \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=1 \
  --model_dir=${MODEL_DIR} \
  --distribution_strategy=tpu \
  --tpu=${TPU_NAME} \
  --steps_per_loop=500


  gcloud compute tpus execution-groups delete bert-tutorial \
  --zone=us-central1-f

  gsutil rm -r gs://tft_free



 gcloud compute ssh bert-tutorial --zone=us-central1-f
