# -*- coding: utf-8 -*-

import argparse
import tqdm
import json
import tensorflow as tf
import subprocess
import re
import os
from pathlib import Path
import time
from absl import logging
logging.set_verbosity("INFO")

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    """Split text to sentences"""
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [' '.join(s.strip().split()) for s in sentences]
    return sentences




MAX_WORDS_PER_SENTENCE = 1000 # 1024 , we keep 1000 considering subwords
BUFFER_EXCEEDED = 100 # If last sentence is more than 500 words, we add it to next line
logging.info("GCP BUCKET ---> {}".format(os.getenv('STORAGE_BUCKET', None)))
STORAGE_BUCKET = os.getenv('STORAGE_BUCKET', None)

def copy_to_gcs_bucket(target_file):
    """Copy to GCS bucket"""
    # Prepare dir
    DATA_DIR = os.path.join(STORAGE_BUCKET, "pile_as_text")
    if tf.io.gfile.exists(DATA_DIR):
        logging.info("{} exists.".format(DATA_DIR))
    else:
        tf.io.gfile.mkdir(DATA_DIR)
    # Copy to GCS bucket
    # tf.io.gfile.copy(target_file, os.path.join(DATA_DIR, target_file))
    parallel_threshold='150M' # minimum size for parallel upload; 0 to disable

    subprocess.check_call([
    'gsutil',
    '-o', 'GSUtil:parallel_composite_upload_threshold={}'.format(parallel_threshold),
    'cp', target_file, os.path.join(DATA_DIR, target_file)
    ])

def write_to_a_file(sentences, file_obj):
    """Write to a file with delimiter and \n"""
    sentences = '__||__'.join(sentences)
    sentences = sentences.replace("\n", "")
    file_obj.write(sentences + " \n")
    
def process(source_file, target_file):
    """Read, process and write"""
    if os.path.exists(target_file):
        raise FileExistsError(target_file)
    
    target_file_obj = open(target_file, "w")
    sentence_per_line = []
    sentence_per_line_counter = 0
    reader = open(source_file, 'r', encoding='utf-8')
    for line_no, line in tqdm.tqdm(enumerate(reader)):
        item = json.loads(line)
        sentences = split_into_sentences(item['text'])
        
        for line in sentences:
            words_per_line = len(line.split())
            sentence_per_line.append(line)
            sentence_per_line_counter += words_per_line
            
            if sentence_per_line_counter >= MAX_WORDS_PER_SENTENCE:
                
                if words_per_line < BUFFER_EXCEEDED:
                    # write to a file
                    # join with delimiter
                    write_to_a_file(sentence_per_line, target_file_obj)
                    # reset
                    sentence_per_line = []
                    sentence_per_line_counter = 0
                else:
                    # If buffer exceeds, we use it for next phase
                    write_to_a_file(sentence_per_line, target_file_obj)
                    sentence_per_line = [line]
                    sentence_per_line_counter = words_per_line
                
    # Delete source file
    time.sleep(2)
    os.remove(source_file)
    # Close target file
    target_file_obj.close()
    copy_to_gcs_bucket(target_file)
    # Delete target file
    time.sleep(2)
    os.remove(target_file)
        
        
def main():
    """Main"""

    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', type=str, metavar='',\
        help="Path of source file jsonl")
    parser.add_argument('-t', '--target_file', type=str, metavar='',\
        help="Path of target file txt")
    # Parse and print the results
    args = parser.parse_args()
    process(args.source_file, args.target_file)
    
if __name__=="__main__":
    main()
