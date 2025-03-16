import re
import csv
import sys

# Ugly but whatever
sys.path.append(".")
from net_default_cfg.default_config import get_default_config
from logger.logger import create_logger
import numpy as np

def one_hot_encoding_type(exp_type):
    types_map = {
        'EQUI' : 5, 
        'OPPO' : 4,
        'SPE1' : 3,
        'SPE2' : 2,
        'SIMI' : 1,
        'REL' : 0
    }

    one_hot = list(np.zeros(len(types_map), dtype = int))
    one_hot[types_map[exp_type]] = 1

    return one_hot
   
def one_hot_encoding_score(value):
    scores_map = {
        '5' : 5, 
        '4' : 4,
        '3' : 3,
        '2' : 2,
        '1' : 1,
        '0' : 0
    }

    one_hot = list(np.zeros(len(scores_map), dtype = int))
    one_hot[scores_map[value]] = 1

    return one_hot

def create_csv(data_files: list[str], file_name: str) -> None:
    alignment_start_pattern = re.compile(r"<alignment>")
    alignment_end_pattern = re.compile(r"</alignment>")
    equality_pattern = re.compile(r"<==>")
    doubleslash_pattern = re.compile(r"//")

    aligment_section = False
    processed_files = []

    for file_path in data_files:
        processed_file = []
        with open(file_path, 'rb') as fin:
            fin.readline()
            for index, line in enumerate(fin):
                processed_line = []
                line = line.decode('latin-1')
                if alignment_end_pattern.match(line):
                    aligment_section = False
                if aligment_section:
                    equality_pos = [m.start() for m in re.finditer(equality_pattern, line)]
                    doubleslash_pos = [m.start() for m in re.finditer(doubleslash_pattern, line)]

                    value = line[doubleslash_pos[1] + 2: doubleslash_pos[2]].strip().strip("\n")
                    explanation = line[doubleslash_pos[0] + 2: doubleslash_pos[1]].strip().strip("\n")
                    first_chunk = line[doubleslash_pos[2] + 2: equality_pos[1]].strip().strip("\n")
                    second_chunk = line[equality_pos[1] + 4:].strip().strip("\n")

                    processed_line.append(first_chunk)
                    processed_line.append(second_chunk)

                    # interesuje nas wszystko poza ALIC i NOALIC - tego nie uwzgledniamy
                    if value != "NIL" and len(explanation) <= 4 and explanation != 'ALIC':  
                        exp = one_hot_encoding_type(explanation)
                        val = one_hot_encoding_score(value)
                        vector = val+exp
                        processed_line.append(vector)
                        processed_file.append(processed_line)

                if alignment_start_pattern.match(line):
                    aligment_section = True

        processed_files += processed_file

    with open(file_name, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['chunk1', 'chunk2', 'vector'])
        writer.writerows(processed_files)


def main():
    train_files = ['semeval_data/train/STSint.gs.headlines.wa',
                   'semeval_data/train/STSint.gs.images.wa',
                   'semeval_data/train_students_answers_2015_10_27.utf-8/STSint.input.answers-students.wa']
    test_files = ['semeval_data/test_goldStandard/STSint.testinput.answers-students.wa',
                  'semeval_data/test_goldStandard/STSint.testinput.headlines.wa',
                  'semeval_data/test_evaluation_task2c/STSint.gs.headlines.wa',
                  'semeval_data/test_evaluation_task2c/STSint.gs.images.wa']
    config = get_default_config()

    output_dir = config.OUTPUT_DIR
    logger = create_logger("Model", output_dir, 0)

    create_csv(train_files, "data/datasets/train.csv")
    logger.info("Created data/datasets/train.csv")
    create_csv(test_files, "data/datasets/test.csv")
    logger.info("Created data/datasets/test.csv")
    create_csv([test_files[0]], "data/datasets/test_answers-students.csv")
    logger.info("Created data/datasets/test_answers-students.csv")
    create_csv([test_files[2]], "data/datasets/test_headlines.csv")
    logger.info("Created data/datasets/test_headlines.csv")
    create_csv([test_files[3]], "data/datasets/test_images.csv")
    logger.info("Created data/datasets/test_images.csv")


if __name__ == '__main__':
    main()

