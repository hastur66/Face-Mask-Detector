#!/usr/bin/env python
# coding: utf-8

#import tensorflow as tf
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
#import tensorflow.keras
#from tensorflow.keras.applications.mobilenet import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2
import os
import time
import imutils
from imutils.video import VideoStream
#import matplotlib.pyplot as plt


print("---> loading face detector model...")
prototxtPath = os.path.sep.join(["./faceModel/deploy.prototxt"])
weightsPath = os.path.sep.join(["./faceModel/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


def detect_and_predict_mask(frame, faceNet):
    # grab the dimensions of the frame creat a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence/probability associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            #face = img_to_array(face)
            #face = preprocess_input(face)
            face = face.astype(np.float32)
            face /= 255.
            
            interpreter = tflite.Interpreter(model_path="Models/mobileNet_model.tflite")
            interpreter.allocate_tensors()

            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            #interpreter.resize_tensor_input(input_details[0]['index'], batch_input.shape) #(batch_size, 512, 512, 3)

            # Adjust output #1 in graph to handle batch tensor
            #interpreter.resize_tensor_input(output_details[0]['index'], batch_input.shape) #(batch_size, 512, 512, 3)
            # Test the model on random input data.

            input_shape = input_details[0]['shape']
            
            face = np.expand_dims(face, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], face)

            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        #for face in faces:
        #face = np.array(face, dtype="float32")
        #preds = maskNet.predict(faces, batch_size=32)
        #faces = np.expand_dims(faces, axis=0)



        preds = output_data


    return (locs, preds)


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # grab the frame from the threaded video stream and resize it to maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and colors use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press `q` key for break the loop/webcam
    if key == ord("q"):
        break

# cleanup
cv2.destroyAllWindows()
vs.stop()





