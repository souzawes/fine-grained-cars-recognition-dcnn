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
# from random_eraser import get_random_eraser
from PathType import PathType

def train(args):

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

    # same shape for all datasets
    input_shape = (args.target_size, args.target_size, 3)

    # labels name
    classes = []
    for _, classe in ds.classes_model.items():
        classes.append(classe)

    # callbacks params
    early_stopper = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss', 
                                    patience=30,
                                    verbose=1,
                                    restore_best_weights=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                    monitor="val_loss",
                                    patience=10,
                                    verbose=1)

    backbone = "ResNet152"

    if args.architecture == 2:
        backbone = "InceptionV3"
    elif args.architecture == 3:
        backbone = "VGG16"
    elif args.architecture == 4:
        backbone = "DenseNet169"
    elif args.architecture == 5:
        backbone = "InceptionResNetV2"

    print(f"\n\t > input shape: {input_shape}")
    print(f"\t > batch size: {batch_size}")
    print(f"\t > dataset: {dataset}")
    print(f"\t > network: {backbone}")
    print(f"\t > n classes: {len(classes)}")
    print(f"\t > learning rate: {args.learning_rate}")

    cnn = CNN(backbone = backbone, classes = classes, input_shape = input_shape)

    cnn.compile(
        weights="imagenet", 
        trainable=False, 
        metrics = [
                'accuracy', 
                tf.keras.metrics.Recall(),
                tfa.metrics.F1Score(num_classes = len(classes)),
                tf.keras.metrics.Precision()
            ],
        learning_rate=args.learning_rate
    )

    print("\n")
    cnn.model.summary()
    print("\n")

    history = cnn.train(
        ds.train_ds,
        ds.val_ds,
        args.epochs,
        10,
        [early_stopper, reduce_lr]
    )

    model_history = pd.DataFrame(history.history)  

    if not os.path.exists("history_training"):
        os.makedirs("history_training")
    
    if not os.path.exists("weights"):
        os.makedirs("weights")

    filename_training = f'{backbone}-{dataset}-{args.epochs}-{args.target_size}x{args.target_size}'

    model_history_filename = f'history_training/{filename_training}'
    model_history.to_csv(f'{model_history_filename}.csv')

    trained_model_weights_path = f"weights/{filename_training}.h5"
    cnn.model.save(trained_model_weights_path)
     
    plot = model_history.plot(y=["val_accuracy", "accuracy", "val_loss", "loss"], figsize=(5,5), xlabel='epochs', title='Train and Validation Learning Curves')
    fig = plot.get_figure()
    fig.savefig(f"{model_history_filename}.png")

    plt.show()

if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-a", "--architecture", help = "Select architecture to train (Options: 1 - ResNet152, 2 - InceptionV3, 3 - VGG16, 4 - DenseNet169, 5 - InceptionResNetV2).", 
    required=True, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("-d", "--dataset", help = "Select dataset to use (Options: 1 - brcars427, 2 - stanford196, 3 - compcars1716).", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("-e", "--epochs", help = "Select quantity of epochs to train (Default: 100).", type=int, default=100)
    parser.add_argument("-ts", "--target_size", help = "Select target size of images used on training (Default: 128).", type=int, default=128)
    parser.add_argument("-bs", "--batch_size", help = "Select batch size used on training (Default: 32).", type=int, default=32)
    parser.add_argument("-dd", "--datasets_directory", help = "Select datasets directory (Default: 'datasets//').", type = PathType(exists=True, type='dir'), default="datasets//")
    parser.add_argument("-lr", "--learning_rate", help = "Select learning rate used on training (Default: 1e-5).", type=float, default=1e-5)
    
    # Read arguments from command line
    args = parser.parse_args()

    start_time = time.monotonic()
    train(args)
    end_time = time.monotonic()
    print(f"\n Training duration: {timedelta(seconds=end_time - start_time)}")
