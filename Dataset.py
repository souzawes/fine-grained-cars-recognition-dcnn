from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class Dataset:
    """
        This class encapsulates all it needed for train a CNN model using ImageDataGenerator. 
        The class will produce a dictionary with the classes with the format {model_id : model_label}, 
        also DataFrameIterators for the training, validation and test sets.

        ...

        Args
        ----------
        path : String
            Path of dataset files.
        target_size : Tuple of integers
            Tuple of integers (height, width). The dimensions to which all images found will be resized.
        batch_size : Int
            Size of the batches of data.
        augmentation_values : Dict
            Dictionary with all possible augmentation values supported by the class ImageDataGenerator from keras 
            (see: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator).
    """

    def __init__(self, path, target_size, batch_size, augmentation_values = {}):      

        # folders containing the training, validation and test sets
        folders = []

        for dataset in os.listdir(path):
            folders.append(os.path.join(path, dataset))

        path_train, path_val, path_test = folders   
        
        
        # dataset name
        dataset = path.replace('datasets/', '')
        
        # path sheet containing informations about dataset
        path_sheet = 'utils/'+ dataset + '.csv'        
        csv_file_classes = []
        with open(path_sheet, 'r', errors='replace') as f:
            csv_file_classes = f.read().splitlines()[1:]
        

        # load id and name of model labels 
        self.classes_model = {}
    
        for line in csv_file_classes:
            id_model, model_label = line.strip().split(',')            
            self.classes_model[id_model] = model_label

        
        # generate images of set train with data augmentation
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            **augmentation_values
        )

        # generate images of set validation with data augmentation
        val_datagen = ImageDataGenerator(                                
            rescale = 1./255
        )    

        # create train dataset
        self.train_ds = train_datagen.flow_from_directory(
                path_train,
                target_size = target_size,
                batch_size =  batch_size,
                class_mode = 'categorical',
                color_mode='rgb',
                shuffle=True
            )
        
        # create validation dataset
        self.val_ds = val_datagen.flow_from_directory(
                path_val,
                target_size = target_size,
                batch_size =  batch_size,
                class_mode = 'categorical',
                color_mode='rgb',
                shuffle=False
            )
        
        # create test dataset
        self.test_ds = val_datagen.flow_from_directory(
                path_test,
                target_size = target_size,
                batch_size =  batch_size,
                class_mode = 'categorical',
                color_mode='rgb',
                shuffle=False
            )