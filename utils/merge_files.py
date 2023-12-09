"""
merge all .txt files at the given directory to a single .txt file
python3 ../utils/merge_files.py --file_dir predictions/eval_sentence_base_all_data_coco_i_ii --file_list generated_sentence_eval_0.txt,generated_sentence_eval_1.txt,generated_sentence_eval_2.txt,generated_sentence_eval_3.txt --output_file_path predictions/eval_sentence_base_all_data_coco_i_ii/generated_sentence_eval.txt
"""
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--file_dir', type=str, required=True)
parser.add_argument('--file_list', type=str, default=None)
parser.add_argument('--output_file_path', type=str, required=True)

args = parser.parse_args()

if __name__ == '__main__':

    if args.file_list is not None:
        file_path_list = [os.path.join(args.file_dir, fn) for fn in args.file_list.split(',')]

    else:
        file_path_list = os.listdir(args.file_dir)
        file_path_list = [os.path.join(args.file_dir, fp) for fp in file_path_list]

    data = []
    for fp in file_path_list:
        with open(fp) as file:
            data += file.readlines()

    with open(args.output_file_path, 'w') as file:
        # random.seed(1)
        # random.shuffle(data)
        for s in data:
            file.write(s)
