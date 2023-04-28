import numpy as np
import tensorflow as tf
import cv2
import os
from script_webcam import detect_and_predict_mask

img_array = np.random.rand(300, 300, 3)
prototxtPath = os.path.sep.join(["./faceModel/deploy.prototxt"])
weightsPath = os.path.sep.join(["./faceModel/res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = tf.keras.models.load_model("./Models/mobileNet_model.h5")

def test_detect_and_predict_mask(img_array, faceNet, maskNet):
    assert detect_and_predict_mask(img_array, faceNet, maskNet) == 0