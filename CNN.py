from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
import numpy as np

class CNN:    
  """
    This class encapsulates a CNN model and uses various methods available by keras. All methods behave similarly to the respective keras methods. 

    ...

    Args
    ----------
    backbone : String
        CNN model.
    input_shape : Tuple of integers
        Shape tuple, it should have exactly 3 inputs channels, and width and height.
    classes : Int
        Number of classes to classify images.
  """

  def __init__(self, backbone = None, input_shape = None, classes = None):
    self.backbone = backbone
    self.input_shape = input_shape
    self.classes = classes

  def compile(self, weights, trainable, metrics = ["accuracy"], learning_rate = 1e-2):
    self.model = Sequential()

    if self.backbone == "ResNet152":
      backbone = ResNet152(input_shape = self.input_shape,
                          weights = weights, 
                          include_top = False)
      
    elif self.backbone == "InceptionV3":
      backbone = InceptionV3(input_shape = self.input_shape,
                             weights = weights, 
                             include_top = False)
      
    elif self.backbone == "VGG16":
      backbone = VGG16(input_shape = self.input_shape,
                       weights = weights, 
                       include_top = False)
      
    elif self.backbone == "DenseNet169":
      backbone = DenseNet169(input_shape = self.input_shape,
                             weights = weights, 
                             include_top = False)

    elif self.backbone == "InceptionResNetV2":
      backbone = InceptionResNetV2(input_shape = self.input_shape,
                                   weights = weights, 
                                   include_top = False)
       
    else:
      print("Invalid backbone")

    for layer in backbone.layers:
      layer.trainable = trainable

    self.model.add(backbone)
    self.model.add(GlobalAveragePooling2D())
    self.model.add(Dense(1024, activation='relu'))
    self.model.add(Dense(512, activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(len(self.classes), activation='softmax'))

    self.model.compile (
        loss = 'categorical_crossentropy',
        optimizer = Adam(learning_rate=learning_rate),
        metrics = metrics
    )

  def train(self, train_ds, val_ds, epochs = 1, workers = 1, callbacks = None, initial_epoch = 0):
    history = self.model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = epochs,
        workers = workers,
        verbose = 1,
        callbacks = callbacks,
        initial_epoch = initial_epoch
    )

    return history

  def predict(self, img):
    # preprocessing image
    # "img" could be a Pillow object
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img / 255.0

    probs = self.model.predict(img)[0]
    prediction = np.argmax(probs)

    image_classification = {'class': self.classes[prediction], 'confidence': (probs[prediction] * 100)}   
      
    return image_classification

  def load(self, model_path):
    self.model = load_model(model_path)

  def save(self, model_path):
    self.model.save(model_path)