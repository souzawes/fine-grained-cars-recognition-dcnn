# Fine-grained Cars Recognition using Deep Convolutional Neural Networks


## π» Project
___

This repository stores an application that performs the training of four architectures of Deep Convolutional Neural Networks using three base sets of different car images, extracting the main metrics (precision, accuracy, f1-score and recall) of each scenario, with the objective of to analyze the performance of architectures in the refined car recognition task

## πSpecifications
___

The sets are divided in ``0: train``, ``1: test`` and ``2: validation``the directories must be structured as the file tree is described in topic [Folders Tree Structure](#π-folders-tree-structure), furthermore, in the `utils` directory, there is a list of labels and id's of each discriminated class . For download of datasets use the links in each section set [Datasets](#π-datasets) 

### πΈ Architectures

- ResNet-152
- Inception-V3
- VGG-16
- Densenet-169

### π Datasets

- [BRCars](https://github.com/danimtk/brcars-dataset)
- [Cars-196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/)

### π Folders Tree Structure 
```
πdatasets
ββββπbrcars
β   ββββπ0
β   β     image1.jgp
β   β     image2.jpg
β   β     ...
β   ββββπ1
β   β     image1.jgp
β   β     image2.jpg
β   β     ...
β   ββββπ2
β         image1.jgp
β         image2.jpg
β         ...
ββββπcompcars
β   ββββπ0
β   ββββπ1
β   ββββπ2
ββββπstanford
    ββββπ0
    ββββπ1
πutils
    brcars427.csv
    compcars1716.csv
    stanford196.csv
```


## π₯ Instalation and dependencies
___

Clone the project

```
git clone https://github.com/souzawes/fine-grained-cars-recognition-dcnn.git
```

```
cd fine-grained-cars-recognition-dcnn
```

We recommend creating a new conda environment from ``environment.yaml`` file:

```
conda env create --name <ENV_NAME> --file=environment.yaml
```

## βΆ Run scripts
___

The main script ```train.py``` can be run using the following command:

```
python train.py
```

### β οΈ Flags Requireds 

The ```-a``` or ```--architecture``` Select architecture to train (Options: 1 - ResNet152, 2 - InceptionV3, 3 - VGG16, 4 - DenseNet169)

The ```-d``` or ```--dataset``` Select dataset to use (Options: 1 - brcars427, 2 - stanford196, 3 - compcars1716)


### π΅ Flags Optional

The ```-e``` or ```--epochs``` Select quantity of epochs to train (Default: 100)

The ```-ts``` or ```--target_size``` Select target size of images used on training (Default: 128)

The ```-bs``` or ```--batch_size``` Select batch size used on training (Default: 32).

The ```-dd``` or ```--datasets_directory``` Select datasets directory (Default: 'datasets//')

The ```-lr``` or ```--learning_rate``` Select learning rate used on training (Default: 1e-5)

## β¨ Future Works

- [ ] Added networks (Inception-v4, Xception and EfficientNet)
- [ ] Implement cross-validation in pipeline
- [ ] Use segmentation techniques to remove background from images to try to improve metric results

