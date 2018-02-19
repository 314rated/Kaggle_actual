#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:23:11 2018

@author: vsl1
"""


from scipy import stats
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from myutils import *
from dataprovider import *
import tensorflow as tf
import random

def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, 112, 112, 1))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate

def augment_image(image, label, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):
    augmented_image = []
    augmented_image_labels = []
    for i in range(0, augementation_factor):
        augmented_image.append(image)
        augmented_image_labels.append(label)
        x= random.randint(1,20) % 3
        if x==0:
            augmented_image.append(tf.contrib.keras.preprocessing.image.random_rotation(image, 180, row_axis=0, col_axis=1, channel_axis=2))
            augmented_image_labels.append(label)
        elif x==1:
            augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(image, -0.1, -0.2, row_axis=0, col_axis=1, channel_axis=2))
            augmented_image_labels.append(label)
        elif x==2:
            augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(image, [0.8, 0.9], row_axis=0, col_axis=1, channel_axis=2))
            augmented_image_labels.append(label)
    
    return np.array(augmented_image), np.array(augmented_image_labels)

def get_minibatch_augmented(imageNames, yData, dataDirectory, processingType, entireDataFlag):
    imageNames = np.array(imageNames)
    shapeOfImage = getImageSize(dataDirectory, imageNames[0])
    
    totalDataPoints = imageNames.shape[0]
    maxBatchSize =  200 if not entireDataFlag else imageNames.shape[0]
    batchSize = min(imageNames.shape[0], maxBatchSize)
    numberOfBatches = int(totalDataPoints/batchSize)
    if totalDataPoints%batchSize !=0:
        numberOfBatches +=1
    

    imagesPerPoint = 2
    b=0
    while True:
        batch_imageNames = []
        batchLabels = []
        
        if b!= numberOfBatches-1:
            batch_imageNames = imageNames[b*batchSize:(b+1)*batchSize]
            batchLabels = yData[b*batchSize:(b+1)*batchSize]
            b = b + 1
        else:
            batch_imageNames = imageNames[b*batchSize:]
            batchLabels = yData[b*batchSize:]
            b = 0
        
        currentBatchSize = len(batch_imageNames)
        mini_x = np.zeros([currentBatchSize*imagesPerPoint, shapeOfImage[0],shapeOfImage[1],shapeOfImage[2]])
        mini_x_processed = np.zeros([currentBatchSize*imagesPerPoint, shapeOfImage[0],shapeOfImage[1],1])
        mini_y = np.zeros([imagesPerPoint*currentBatchSize])
        mini_x_imageNames_list  = []
        for batchpointindex in range(currentBatchSize):
            path = dataDirectory + batch_imageNames[batchpointindex]
            image, processedImage = preprocessImage(path, processingType)
            labelsToAugment = np.array(int(batchLabels[batchpointindex]))
            imagesToAugment = processedImage
            aug_X, aug_y= augment_image(imagesToAugment, labelsToAugment,
                                       augementation_factor=1,
                                       use_random_rotation=True,
                                       use_random_shift=True, 
                                       use_random_zoom=True)
            mini_x[batchpointindex*imagesPerPoint: (batchpointindex+1)*imagesPerPoint] = image
            mini_x_processed[batchpointindex*imagesPerPoint: (batchpointindex+1)*imagesPerPoint] = aug_X
            mini_y[batchpointindex*imagesPerPoint: (batchpointindex+1)*imagesPerPoint] = aug_y
            for j in range(imagesPerPoint):   
                mini_x_imageNames_list.append(batch_imageNames[batchpointindex])
                
        mini_x_imageNames = np.array(mini_x_imageNames_list)
        yield (mini_x,mini_x_processed,mini_y,numberOfBatches,mini_x_imageNames)
        
 