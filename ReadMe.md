# Face Mask Detector using CNN

## Requirments

Following python frameworks and libraries are needed to train and deploy the system. Running the Jupyter notebooks in Anaconda-Conda environment in local computer with following requirments or conda environment provided by Kaggle is sutible. Kaggle environment is much more sutible since the requirments are pre-installed and configured as well as dataset paths are set default as in Kaggle (change the file paths according to your local device when training the model).
* Tensorflow
* Keras
* OpenCV
* Numpy, Pandas, Sklearn, Matplotlib
* os, time, argmax, imutils, PIL

## Dataset

The dataset is an open [dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) published in Kaggle.com with 12000 mask and maskless image data.

## Face model

Firts to detect the faces pretrained [caffe](https://caffe.berkeleyvision.org) model is used. Once the face model detects a face it is extracted and send to classifie with the face mask model.

## Face Mask model

Face Mask model classifie whether detected face is wearing mask or not and make prediction. Face mask model use either MobileNet, MobileNetV2, or EfficientNet architecture to train the model. Trained models are saved in .h5 format.

## Running the web cam

Using trained models and openCV the web cam detects and predict, whether individual wears mask or not. 