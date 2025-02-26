import argparse
import os
import pickle
import random
from pathlib import Path
from tqdm import tqdm

import utilities.processor as midi_processor
from utilities.constants import seed_everything

TRAIN_SIZE, TEST_SIZE, VALIDATION_SIZE = 0.8, 0.1, 0.1


# prep_midi
def prep_custom_midi(custom_root, output_dir):
    """
    ----------
    Author: Damon Gwinn, modified Huw Cheston
    ----------
    Pre-processes custom data
    ----------
    """

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    midi_files = [str(i) for i in Path(custom_root).glob("*.mid")]
    for f in midi_files:
        assert os.path.isfile(f)

    print("Found", len(midi_files), "pieces")
    print("Preprocessing...")

    total_num_files = len(midi_files)
    num_files_valid = round(total_num_files * 0.15)
    num_files_test = round(total_num_files * 0.15)
    random.shuffle(midi_files)
    midi_paths_valid = midi_files[:num_files_valid]
    midi_paths_test = midi_files[num_files_valid:num_files_valid + num_files_test]
    midi_paths_train = midi_files[num_files_valid + num_files_test:]

    n_train, n_valid, n_test = 0, 0, 0

    for files_paths, split_type in (
            (midi_paths_train, "train"), (midi_paths_valid, "validation"), (midi_paths_test, "test")
    ):
        for mid in tqdm(files_paths, desc=split_type):
            f_name = mid.split("/")[-1] + ".pickle"

            if split_type == "train":
                o_file = os.path.join(train_dir, f_name)
                n_train += 1
            elif split_type == "validation":
                o_file = os.path.join(val_dir, f_name)
                n_valid += 1
            elif split_type == "test":
                o_file = os.path.join(test_dir, f_name)
                n_test += 1
            else:
                print("ERROR: Unrecognized split type:", split_type)
                return False

            prepped = midi_processor.encode_midi(mid)

            o_stream = open(o_file, "wb")
            pickle.dump(prepped, o_stream)
            o_stream.close()

    print("Num Train:", n_train)
    print("Num Val:", n_valid)
    print("Num Test:", n_test)

    return True


# parse_args
def parse_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Parses arguments for preprocess_midi using argparse
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("root", type=str, help="Root folder for the custom data.")
    parser.add_argument("-output_dir", type=str, default="./dataset/e_piano", help="Output folder to put the preprocessed midi into.")

    return parser.parse_args()

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Preprocesses maestro and saved midi to specified output folder.
    ----------
    """

    args = parse_args()
    root = args.root
    output_dir = args.output_dir

    print("Preprocessing midi files and saving to", output_dir)
    prep_custom_midi(root, output_dir)

    print("Done!")
    print("")


if __name__ == "__main__":
    seed_everything()
    main()
