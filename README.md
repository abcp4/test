# LearnGeo

Learn + Cenpes Research Repository.

Seismogram Quality Prediction, based on the Miniception Network.

##
### 1 Setup Configuration
 
Build the Project Docker Image

```
sh.build
```

Run the Project Docker Image

```
sh.run
```

Dockerfile should be eddited depending on your machine configuration. 
It utilizes a 1.14.1 tensorflow Image.
##
### 2 Load Data
Inside the container use gdown command to download dataset from Google Drive
```
gdown https://drive.google.com/uc?id=14pTmqBrZwOabhleavzp2fu1zHzXBOs0z
```
Extract it in your root folder
```
chmod +x dataset.zip
```
##
### 3 Experimentation
All the possible configuration are manipulated by the main.py file, 
that is located on the root from this project. 
You can see the options by typing the following command.
```
python3 main.py -h
```
You can also run multiple tasks by running model.sh bash file.
```
sh model.sh
```
#
### Options 
  #### Basic Configurations
  
The basic network and GPU configurations are stored on the folder named 'config', 
placed on the root from this project.

-baseline_config : Choose the baseline initial configuration of hyperparameters.
By default, it chooses the file baseline.json.

-config : Choose the miniception initial configuration of hyperparameters.
By default, it chooses the file miniception_R17.json.

-gpu_config : Choose the baseline initial configuration of hyperparameters.
By default, it chooses the file gpu.json.

   #### Model Selection
   
-baseline : chose the baseline model to run. 
All the models perform feature extraction in each layer and 
    apply it's values on SVM.
You can choose between vgg16 vgg19 inceptionV3.
    
-miniception : choose the Miniception Model to perform a Cross-Validation.
The miniception model has several variations in the default architecture, 
    that are named from A to D. Miniception A is the default architecture, 
    and miniception D_ordinal is the best architecture found.   
You can choose between A,B,C,D, D_ordinal

-holdout : choose the Miniception Model to perform a Holdout.
In Holdout the algorithm trains on dataset 1 
    and predict the values for dataset 2.
You can choose between A,B,C,D, D_ordinal

-evaluate: Restore the last checkpoint, train on new data aquisition and
    update the network weigths. Return the predictions for the given data.
You can choose between A,B,C,D, D_ordinal

  #### Fine Tunning Configurations 

-slice : Select a subset from the Train Dataset.
It must be a value between 0.0 and 1.0 to select fractions from the data or 
    an integer to fix the train amount.
By default its values is set on 1.0, then select the whole Train data.

-num_epochs : Number of Epochs.
Must be an integer. By default is set to 100.

-batch : The size of the Batch.
Must be an integer. By default is set to 1.

-lr : The Learning Rate to perform train.
Must be an float. By default is set to 5e-5.

-alpha : Multiply the number of filters in each convolution.
Must be an integer. By default is set to 1.

-width : Width of the image.
Must be an integer or None.
By default is set to None, to accept images from variable size.

-height : Height of the image.
Must be an integer or None. 
By default is set to None, to accept images from variable size.

-beta : Divide the heigth and the width of the image.
Must be an integer.
By default is set to 1, to main the image proportions. 

-n_blocks : Number of Blocks in the Miniception Architecture.
Must be an integer. By default is set to 4 blocks.

-aug : Define if want to augment data or not.
Must be a boolean. 
By Default is set to False. Can be set to True to augment data.

-build_aug : Define weather or not to save images on the directory.
In the begining from algorithm it save the images.
Must have images saved to perform data augmentation.
Must be a boolean. 
By Default is set to False. Can be set to True to save augmentation.
                    
-aug_images : Number of Augmented Images in each transformation.
Must be an integer. By default is set to 200.

  #### Examples

You can run default configurations in the simplest way
```
python3 main.py -miniception D_ordinal
```
It's possible to change between models like this. 
Also, it's available to run multiple models at one command
```
python3 main.py -baseline vgg16
python3 main.py -holdout A
python3 main.py -evaluate D_ordinal
python3 main.py -baseline vgg16 vgg19 inceptionV3
```
You can fine-tune models like this
```
python3 main.py -miniception D -alpha 8 -lr 1e-6 -num_epochs 40
python3 main.py -holdout D -aug true -aug_images 50
```
You can change the basic configuration
```
python3 main.py -holdout D -aug true -aug_images 100 -config miniception_R49
```
