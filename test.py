import argparse
import tensorflow_addons as tfa
import tensorflow as tf
import time
from datetime import timedelta
from Dataset import Dataset
from CNN import CNN
import os
import pandas as pd 
import matplotlib.pyplot as plt
from random_eraser import get_random_eraser
from tensorflow.keras.models import load_model
from PathType import PathType
from pathlib import Path

def test(args):
    
    print("")

    # samples used a each iteration
    batch_size = args.batch_size
        
    # params of data augmentation
    augmentation_values = {
        'shear_range' : 0.15,
        'zoom_range' : 0.2,
        'horizontal_flip' : True,
        # 'preprocessing_function': get_random_eraser(v_l=0, v_h=1),
        'height_shift_range' : 0.2,
        'width_shift_range' : 0.2,
        'rotation_range': 30
    }

    dataset = "brcars427"

    if args.dataset == 2:
        dataset = "stanford196"
    elif args.dataset == 3:
        dataset = "compcars1716"


    # path location of dataset
    path_dataset = args.datasets_directory + dataset

    # dataset
    ds = Dataset(
        path_dataset,
        (args.target_size, args.target_size),
        batch_size,
        augmentation_values
    )

    print(f"\n\t > batch size: {batch_size}")
    print(f"\t > dataset: {dataset}")
    _set = ds.test_ds
    _set_description = "test"

    if args.set == 2:
        _set = ds.val_ds
        print(f"\t > Validation set selected\n")
        _set_description = "val"
    elif args.set == 3:
        _set = ds.train_ds
        print(f"\t > Train set selected\n")
        _set_description = "train"
    else:
        print(f"\t > Test set selected\n")

    trained_model = load_model(args.filename)
    trained_model.summary()

    print("")
    history_evaluation = trained_model.evaluate(_set)
    evaluate_history = pd.DataFrame([history_evaluation], columns=["loss", "accuracy", "recall", "f1_score", "precision"])

    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    evaluate_history_path = f'metrics/{Path(args.filename).stem}-{_set_description}.csv'
    evaluate_history.to_csv(evaluate_history_path)
    print(f"Metrics saved at: {evaluate_history_path}")
    

if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Initialize parser
    parser = argparse.ArgumentParser()

    # python test.py -d 1 -f weights/brcars/ResNet152-brcars427-100-128x128.h5 -s 1
    # python test.py -d 3 -f weights/compcars/ResNet152-compcars1716-100-128x128.h5 -s 1
    # python test.py -d 3 -f weights/ResNet152-stanford196-100-128x128.h5 -s 1
    
    # Adding optional argument
    parser.add_argument("-d", "--dataset", help = "Select dataset to use (Options: 1 - brcars427, 2 - stanford196, 3 - compcars1716).", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("-f", "--filename", help = "Select weights path. Metrics i'll be saved on metrics folder.", type = PathType(exists=True, type='file'), required=True)
    parser.add_argument("-dd", "--datasets_directory", help = "Select datasets directory (Default: 'datasets//').", type = PathType(exists=True, type='dir'), default="datasets//")
    parser.add_argument("-ts", "--target_size", help = "Select target size of images used on training (Default: 128).", type=int, default=128)
    parser.add_argument("-bs", "--batch_size", help = "Select batch size used on training (Default: 32).", type=int, default=32)
    parser.add_argument("-s", "--set", help = "Select which set will be used (Options: 1 - test set, 2 - val test, 3 - train set)", type=int, default=1, choices=[1, 2, 3])
    
    # Read arguments from command line
    args = parser.parse_args()

    start_time = time.monotonic()
    test(args)
    end_time = time.monotonic()
    print(f"\n Test duration: {timedelta(seconds=end_time - start_time)}")
