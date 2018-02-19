# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 07:59:58 2018

@author: Surbhi
"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def mag(x): 
    import math
    return math.sqrt(sum(i**2 for i in x))


def addBiasRow(data):
    numberOfDataPoints = data.shape[0]
    onesRow=np.ones([numberOfDataPoints,1]) # add 1 for each feature => make a row of 1s
    dataWithBiasRow = np.concatenate((onesRow,data),axis=1)
    return dataWithBiasRow


def addPowerOfFeature(column, degree, data):
    requiredFeature = data[:,column]
    columnVectorOfFeatureRaisedToPower= toColumnVector(requiredFeature**degree)
    dataWithBiasRow = np.column_stack((data, columnVectorOfFeatureRaisedToPower))
    return dataWithBiasRow
    
def getModelVariables(X):
    numberOfFeaturesWithBias = X.shape[1]
    theta = np.ones([numberOfFeaturesWithBias,1])
    return(X, theta)

def toColumnVector(oneDarray):
    if(len(oneDarray.shape)>1):
        return oneDarray
    return np.array(oneDarray).reshape(oneDarray.shape[0],1)

def extractColumns(X, modelNumber):
    if modelNumber==1:
        return toColumnVector(X[:, 0])
    if modelNumber==2:
        return toColumnVector(X[:,1])
    else:
        return X# Block for finding solution and cost function

def trainTestSplit(X, y, testSize):
    import random
    N = X.shape[0]
    samplesInValidation = int(N*testSize)
    listOfDataPointIndices = np.arange(N)
    validationSetDataPoints = random.sample(range(0, N), samplesInValidation)
    trainSetDataPoints = list(set(listOfDataPointIndices) - set(validationSetDataPoints))
    XtrainingData = X[trainSetDataPoints,:]
    yTrainingData = y[trainSetDataPoints]
    XvalidationData = X[validationSetDataPoints,:]
    yValidationData = y[validationSetDataPoints]
    return (XtrainingData, yTrainingData, XvalidationData, yValidationData)


def toOneHotEncoding(y, minValue, maxValue):
    length =  maxValue - minValue + 1
    vector = np.zeros(length)
    indexOfY  = int(y-minValue)
    vector[indexOfY] = 1
    return vector

def toOneHotEncodingVectorForm(y, minValue, maxValue):
    length =  maxValue - minValue + 1
    result = np.zeros([y.shape[0], length])
    for i in range(y.shape[0]):
        vector = np.zeros(length)    
        indexOfY  = int(y[i]-minValue)
        vector[indexOfY] = 1
        result[i,:]=vector
    return result
    
def getNextBatch(batchSize, batchNumber, X, y):
    return X[batchNumber*batchSize: (batchNumber+1)*batchSize], y[batchNumber*batchSize: (batchNumber+1)*batchSize]

def doKFoldCrossValidation(K, modelNumber, X, y, X_train):
    N = X_train.shape[0]
    samplesPerFold = int(N/K)
    listOfDataPointIndices = np.arange(N)
    
    totalError = 0
    for i in range(K):
        validationSetDataPoints = np.arange(i*samplesPerFold, (i+1)*samplesPerFold)
        trainSetDataPoints = [x for x in listOfDataPointIndices if(x not in validationSetDataPoints)]
        XtrainingData = X[trainSetDataPoints,:]
        yTrainingData = y[trainSetDataPoints]
        
        XvalidationData = X[validationSetDataPoints,:]
        yValidationData = y[validationSetDataPoints]
        
        (Xtrain, theta) =  getModelVariables(extractColumns(XtrainingData, modelNumber))    
        theta = getClosedFormSolution(yTrainingData, theta, Xtrain)
        totalError += costFunction(yValidationData, addBiasRow(extractColumns(XvalidationData, modelNumber)), theta)
    return(totalError*1.0/K)

alongYAxis = 0
alongXAxis = 1

def get_minibatch(training_x, training_y):
    ## Read generator functions if required.
    batchSize = min(training_x.shape[0], 1000)
    numberOfBatches = int(training_x.shape[0]/batchSize)
    for b in range(numberOfBatches):
        mini_x = training_x[b*batchSize:(b+1)*batchSize]
        mini_y = training_y[b*batchSize:(b+1)*batchSize]
        yield mini_x,mini_y


"""global_step = tf.Variable(0, trainable=False)
learning_rate = ALPHA
decay_steps = 200
decay_rate = 0.0001
learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate)

# Passing global_step to minimize() will increment it at each step.
optimizer = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(cost, global_step=global_step)
)

optimizer = (
        tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5)
        .minimize(cost, global_step=global_step)
        )"""

