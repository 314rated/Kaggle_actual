# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:42:32 2018

@author: Surbhi
"""
###############################################################################
from scipy import stats
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from myutils import *
import tensorflow as tf
from sklearn.utils import shuffle
###############################################################################
dataDirectory = './dataset/'
segmentedImagesPath = './train/train/'
alongYAxis = 0
alongXAxis = 1
ALPHA=1e-5
###############################################################################
def uploadInformation():
    #test_list: This file contains a list of image names for test dataset.
    test_list = pd.read_csv('test_list.txt',header=None).as_matrix().flatten()#[:testingSize]
    
    #train_list: This file contains a list of image names and the corresponding labels for training dataset
    train_list = pd.read_csv('train_list.txt',header=None, delimiter=' ')
    #train_list = shuffle(train_list)
    train_list=train_list.as_matrix()
    
    
    #class_index_map: This file contains the label to index mapping for the 20 classes.
    class_index_map = pd.read_csv('class_index_map.txt', header=None, index_col=None)
    class_index_map[:][0] = class_index_map[:][0].str[1:-1]
    class_index_map = class_index_map.as_matrix()
    
    class_index_map_dict = {}
    for index in range(class_index_map.shape[0]):
        key  = class_index_map[index,1]
        value = class_index_map[index,0]
        class_index_map_dict[key] = value
    #--------------------------------------------------------------------------
    print(class_index_map_dict)
    return (train_list, test_list, class_index_map, class_index_map_dict)


def getImageSize(dataDirectory, trainImageName):
    path = dataDirectory + trainImageName
    sampleImage=plt.imread(path)
    return sampleImage.shape


def getNameLists(train_list, test_list, dataDirectory):
    trainImageNames = train_list[:,0]
    testImageNames = test_list
    return (trainImageNames, testImageNames)

def preprocessImage(path, processingType):
    img = cv2.imread(path,cv2.COLOR_BGR2RGB)
    img2 = np.copy(img)
    
    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray = denoise(gray)
    
    if processingType=='edge':
        from skimage.feature import canny
        edges = canny(gray/255.)
        edges = edges.reshape(edges.shape[0], edges.shape[1],1)
        result = edges
    elif processingType == 'lap':
        #ret, thresholdedImage = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        lap = cv2.Laplacian(gray,cv2.CV_64F)
        #laplacian=edgedetect(laplacian).astype(np.uint8)
        #laplacian = np.max( np.array([ edgedetect(img[:,:, 0]), edgedetect(img[:,:, 1]), edgedetect(img[:,:, 2]) ]), axis=0 )
        #return img, edges.reshape(edges.shape[0], edges.shape[1],1)
        lap = lap.reshape(lap.shape[0], lap.shape[1],1)
        result = lap
    else:
        result = gray.reshape(gray.shape[0], gray.shape[1],1)
        
    return img, result

def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255
    return sobel


def testfun(id):
    path = dataDirectory + trainImageNames[id]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    e=edgedetect(laplacian).astype(np.uint8)
    #ret, thresholdedImage = cv2.threshold(e,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    plt.imshow(e.astype(np.uint8), cmap='gray')
    



def denoise(frame):
    frame = cv2.medianBlur(frame,3)
    frame = cv2.GaussianBlur(frame,(1,1),0)    
    return frame

def testfun2(id):
    path = dataDirectory + trainImageNames[id]
    a,b=preprocessImage(path, 'lap')
    plt.imshow(b.reshape(112,112))
    plt.show()
    plt.imshow(a[:,:,1], cmap='gray')
    plt.show()

##################### Data Ingestion ##########################################
(train_list, test_list, class_index_map, class_index_map_dict) = uploadInformation()
(trainImageNames, testImageNames) = getNameLists(train_list, test_list, dataDirectory)
imageShape = getImageSize(dataDirectory, train_list[:,0][0])
classNames = class_index_map[:,0]
classIds  = class_index_map[:,1]

yTrainEntire = train_list[:,1]

numberOfDataPoints = train_list.shape[0]
numberOfTrainData =  int(numberOfDataPoints * 0.75)

yTrain = yTrainEntire[:numberOfTrainData]
trainImages = trainImageNames[:numberOfTrainData]
yVal = yTrainEntire[numberOfTrainData:]
valImageNames = trainImageNames[numberOfTrainData:]

def mapPredictedValuesToLabels(predictedValues, dictionary):
    result = []
    for i in range(len(predictedValues)):
        result.append(dictionary[predictedValues[i]])
    return result

def get_minibatch(imageNames, yData, dataDirectory, entireData):
    ## Read generator functions if required.
    imageNames = np.array(imageNames)
    maxBatchSize = 100 if not entireData else imageNames.shape[0]
    batchSize = min(imageNames.shape[0], maxBatchSize)

    shapeOfImage = getImageSize(dataDirectory, imageNames[0])
    mini_x = np.zeros([batchSize, shapeOfImage[0],shapeOfImage[1],shapeOfImage[2]])
    #mini_x_processed = np.zeros([batchSize, shapeOfImage[0],shapeOfImage[1],1])
    numberOfBatches = int(imageNames.shape[0]/batchSize)
    b=0
    
    while True:
        mini_x_imageNames = imageNames[b*batchSize:(b+1)*batchSize]
        for index in range(batchSize):
            path = dataDirectory + mini_x_imageNames[index]
            mini_x[index] = preprocessImage(path)
        mini_y = yData[b*batchSize:(b+1)*batchSize]
        b = (b+1)%(numberOfBatches) 
        yield (mini_x,mini_y,numberOfBatches,mini_x_imageNames)

###############################################################################

def get_minibatch_processed(imageNames, yData, dataDirectory, processingType, entireDataFlag):
    ## Read generator functions if required.
    
    imageNames = np.array(imageNames)
    shapeOfImage = getImageSize(dataDirectory, imageNames[0])
    
    totalDataPoints = imageNames.shape[0]
    maxBatchSize = 200 if not entireDataFlag else imageNames.shape[0]
    batchSize = min(imageNames.shape[0], maxBatchSize)  
    numberOfBatches = int(totalDataPoints/batchSize)
    if totalDataPoints%batchSize !=0:
        numberOfBatches +=1
    
    b=0
    while True:
        batch_imageNames = []
        batchLabels  = []
        if b== numberOfBatches-1:
            batch_imageNames = imageNames[b*batchSize:]
            batchLabels = yData[b*batchSize:]
            b = 0
        else:
            batch_imageNames = imageNames[b*batchSize:(b+1)*batchSize]
            batchLabels = yData[b*batchSize:(b+1)*batchSize]
            b +=1
            
        
        currentBatchSize = len(batch_imageNames)
        mini_x = np.zeros([currentBatchSize, shapeOfImage[0],shapeOfImage[1],shapeOfImage[2]])
        mini_x_processed = np.zeros([currentBatchSize, shapeOfImage[0],shapeOfImage[1],1])
        mini_y = []
        for batchpointindex in range(len(batch_imageNames)):
            path = dataDirectory + batch_imageNames[batchpointindex]
            image, processedImage = preprocessImage(path, processingType)
            mini_x_processed[batchpointindex] = processedImage
            mini_x[batchpointindex] = image
            
        mini_y  = np.array(batchLabels)
        yield (mini_x,mini_x_processed,mini_y,numberOfBatches,batch_imageNames)


def get_minibatch_testdata(imageNames, dataDirectory, processingType, entireDataFlag):
    ## Read generator functions if required.
    
    imageNames = np.array(imageNames)
    shapeOfImage = getImageSize(dataDirectory, imageNames[0])
    
    totalDataPoints = imageNames.shape[0]
    maxBatchSize = 200 if not entireDataFlag else imageNames.shape[0]
    batchSize = min(imageNames.shape[0], maxBatchSize)  
    numberOfBatches = int(totalDataPoints/batchSize)
    if totalDataPoints%batchSize !=0:
        numberOfBatches +=1
    
    b=0
    while True:
        batch_imageNames = []
        if b== numberOfBatches-1:
            batch_imageNames = imageNames[b*batchSize:]
            b = 0
        else:
            batch_imageNames = imageNames[b*batchSize:(b+1)*batchSize]
            b +=1
            
        
        currentBatchSize = len(batch_imageNames)
        mini_x = np.zeros([currentBatchSize, shapeOfImage[0],shapeOfImage[1],shapeOfImage[2]])
        mini_x_processed = np.zeros([currentBatchSize, shapeOfImage[0],shapeOfImage[1],1])
        for batchpointindex in range(len(batch_imageNames)):
            path = dataDirectory + batch_imageNames[batchpointindex]
            image, processedImage = preprocessImage(path, processingType)
            mini_x_processed[batchpointindex] = processedImage
            mini_x[batchpointindex] = image
            
        yield (mini_x,mini_x_processed,numberOfBatches,batch_imageNames)


