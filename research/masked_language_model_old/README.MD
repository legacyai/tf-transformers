
# Prepare Data

python3 1_data_to_text.py data.name=wikipedia  data.version=20200501.en data.output_text_file=/home/sidhu/Datasets/data/wikipedia.txt
python3 1_data_to_text.py data.name=bookcorpus  data.output_text_file=/home/sidhu/datasets/bookcorpus.txt

# Prepare tfrecords

nohup python3 2_text_to_features.py tokenizer.model_file_path=/home/sidhu/Datasets/data/t5_extended_vocab/new_spiece.model     tokenizer.do_lower_case=false     data.tfrecord_output_dir=/home/sidhu/Datasets/data/wiki_tfrecords     data.tfrecord_filename=wiki     data.tfrecord_nfiles=10     data.input_text_files=[/home/sidhu/Datasets/data/wikipedia.txt]     data.batch_size=1024 > wiki_tfrecord.log &


Bookcorpus

nohup python3 2_text_to_features.py tokenizer.model_file_path=/home/sidhu/Datasets/data/t5_extended_vocab/new_spiece.model     tokenizer.do_lower_case=false     data.tfrecord_output_dir=/home/sidhu/Datasets/data/bookcorpus_tfrecords     data.tfrecord_filename=bookcorpus     data.tfrecord_nfiles=10     data.input_text_files=[/home/sidhu/Datasets/data/bookcorpus.txt]     data.batch_size=1024 > bookcorpus_tfrecord.log &


python3 train_mlm.py    data.tfrecord_path_list=["/home/sidhu/Datasets/bookcorpus_tfrecords", "/home/sidhu/Datasets/wiki_tfrecords"] \
    tokenizer.model_file_path=/home/Sidhu/Datasets/vocab/new_spiece.model

python3 train_mlm.py   tokenizer.model_file_path=/home/sidhu/Datasets/vocab/new_spiece.model \
                       model.model_save_dir=/home/sidhu/Projects/joint_bert
