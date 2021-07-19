
# Prepare Data

python3 1_data_to_text.py data.name=wikipedia  data.version= data.output_text_file=/home/Sidhu/datasets/wikipedia.txt
python3 1_data_to_text.py data.name=bookcorpus  data.output_text_file=/home/Sidhu/datasets/bookcorpus.txt

# Prepare tfrecords

nohup python3 2_text_to_features.py tokenizer.model_file_path=/home/sidhu/Datasets/vocab/new_spiece.model     tokenizer.do_lower_case=false     data.tfrecord_output_dir=/home/sidhu/Datasets/wiki_tfrecords     data.tfrecord_filename=wiki     data.tfrecord_nfiles=25     data.input_text_files=[/home/sidhu/Datasets/wikipedia.txt]     data.batch_size=1024 > wiki_tfrecord.log &


Bookcorpus

nohup python3 2_text_to_features.py tokenizer.model_file_path=/home/sidhu/Datasets/vocab/new_spiece.model     tokenizer.do_lower_case=false     data.tfrecord_output_dir=/home/sidhu/Datasets/bookcorpus_tfrecords     data.tfrecord_filename=bookcorpus     data.tfrecord_nfiles=10     data.input_text_files=[/home/sidhu/Datasets/bookcorpus.txt]     data.batch_size=1024 > bookcorpus_tfrecord.log &
