# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from myutils import *
from dataprovider import *
from augment import *
import inception
from inception import transfer_values_cache
from mytensorlayers import *
tf.reset_default_graph()

import time
timestr = time.strftime("%Y%m%d-%H%M%S")

typeOfFile = "_all3_"+timestr
logFile = './checkpoints/'+typeOfFile +'.txt'

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
session = tf.Session(config = config)

trainYielder = get_minibatch_processed(trainImages, yTrain, dataDirectory, 'gray', entireDataFlag=False)
valYielder = get_minibatch_processed(valImageNames, yVal, dataDirectory, 'gray', entireDataFlag=False)
testYielder = get_minibatch_testdata(testImageNames, dataDirectory, 'gray', entireDataFlag=False)

#augmentationYielder = get_minibatch_augmented(trainImages, yTrain, dataDirectory, entireDataFlag=False)

edgeYielder = get_minibatch_augmented(trainImages, yTrain, dataDirectory, 'edge', entireDataFlag=False)
lapYielder = get_minibatch_augmented(trainImages, yTrain, dataDirectory, 'lap',entireDataFlag=False)
grayYielder = get_minibatch_augmented(trainImages, yTrain, dataDirectory,'gray' ,entireDataFlag=False)


val_edgeYielder = get_minibatch_processed(valImageNames, yVal, dataDirectory, 'edge', entireDataFlag=False)
val_lapYielder = get_minibatch_processed(valImageNames, yVal, dataDirectory, 'lap',entireDataFlag=False)


def getFileName(train, imageName):
    return './pickles/inception_mydata_train_'+imageName +'.pkl'


def writeImageValues():
    print("Processing Inception transfer-values for training-images ...")
    batchCount = 0
    while(True):
        trainBatchX, _, trainBatchY,numberOfBatches,namesTrain=next(trainYielder)
        x_batch, y_true_batch = trainBatchX, toOneHotEncodingVectorForm(trainBatchY.flatten(),0,19)
        for i in range(trainBatchX.shape[0]):
            np.array(transfer_values_cache(cache_path=getFileName(True,namesTrain[i]) , images=np.expand_dims(x_batch[i],axis=0), model=model))
        batchCount += 1
        if batchCount == numberOfBatches:
            break
    
    
    batchCount = 0
    while(True):
        valX, _, valY, numberOfBatches2, namesVal=next(valYielder)
        x_valid_batch, y_valid_batch = valX, toOneHotEncodingVectorForm(valY.flatten(),0,19)        
        for i in range(valX.shape[0]):
            np.array(transfer_values_cache(cache_path=getFileName(False,namesVal[i]), images=np.expand_dims(x_valid_batch[i],axis=0), model=model))
        batchCount += 1
        if batchCount == numberOfBatches2:
            break    

    batchCount = 0
    while(True):
        testX,_,numberOfBatches3,testXImageNames=next(testYielder)
        for i in range(testX.shape[0]):
            np.array(transfer_values_cache(cache_path=getFileName(False,testXImageNames[i]), images=np.expand_dims(testX[i],axis=0), model=model))
        batchCount += 1
        if batchCount == numberOfBatches3:
            print("done saving data")
            break    
    return



#session = tf.Session()
inception.maybe_download()
model = inception.Inception()
transfer_len = int( model.transfer_len)
inceptionFeatureVec = tf.placeholder(tf.float32, shape=[None, transfer_len], name='incepFV')

##Network graph params
num_input_channels = 1
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

filter_size_conv4 = 3
num_filters_conv4 = 128

fc_layer_size = 256
num_classes = 20

processedImages = tf.placeholder(tf.float32, shape=[None, 112,112,num_input_channels], name='procImages')
layer_conv1, weights_conv1 = create_convolutional_layer(input=tf.cast(processedImages, tf.float32),
               num_input_channels=num_input_channels,
               conv_filter_size=filter_size_conv1,
               use_batchNorm = False,
               num_filters=num_filters_conv1,
               name='myConv1')

pool1 = tf.layers.max_pooling2d(inputs=layer_conv1, pool_size=[2, 2], strides=2)

layer_conv2, weights_conv2 = create_convolutional_layer(input=pool1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               use_batchNorm = False,
               num_filters=num_filters_conv2,
               name='myConv2')

pool2 = tf.layers.max_pooling2d(inputs=layer_conv2, pool_size=[2, 2], strides=2)

layer_conv3, weights_conv3 = create_convolutional_layer(input=pool2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               use_batchNorm = False,
               num_filters=num_filters_conv3,
               name='myConv3')

layer_conv4, weights_conv4 = create_convolutional_layer(input=layer_conv3,
               num_input_channels=num_filters_conv3,
               conv_filter_size=filter_size_conv4,
               use_batchNorm = False,
               num_filters=num_filters_conv4,
               name='myConv4')
pool3 = tf.layers.max_pooling2d(inputs=layer_conv4, pool_size=[2, 2], strides=2)


layer_flat = create_flatten_layer(pool3)

layer_fc1, weights_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_batchNorm = False,
                     use_relu=True)

dropout_layer_fc1 = tf.layers.dropout(layer_fc1, rate=0.8, training=True, name="mydropout")
concatenatedFeature = tf.concat([dropout_layer_fc1, inceptionFeatureVec], axis=1)

layer_fc2, weights_fc2 = create_fc_layer(input=concatenatedFeature,
                     num_inputs=fc_layer_size + transfer_len,
                     num_outputs=num_classes,
                     use_batchNorm = False,
                     use_relu=False) 




y_true = tf.placeholder(tf.float32, shape=[None, 20], name='y_true')
y_true_cls = tf.argmax(y_true, axis=alongXAxis)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.argmax(y_pred, axis=alongXAxis)

session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

regularizerLoss_fc = tf.nn.l2_loss(weights_fc1) + tf.nn.l2_loss(weights_fc2)
regularizerLoss_conv = tf.nn.l2_loss(weights_conv1) + tf.nn.l2_loss(weights_conv2) + tf.nn.l2_loss(weights_conv3) #+ tf.nn.l2_loss(weights_conv4)
regularizerLoss = regularizerLoss_fc + 0.1*regularizerLoss_conv 
cost = tf.reduce_mean(cross_entropy + 0.02*regularizerLoss)



optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer()) 


saver = tf.train.Saver()
writer = tf.summary.FileWriter("./output", session.graph)


def predictTest(inceptionModel, testDataProvider):
    batchCount = 0
    finalList  = []
    imageNames = []
    while(True):
        valX, valXGray, numberOfBatches, namesTest = next(testDataProvider)
        for i in range(valX.shape[0]):
            transfer_values_val = np.array(transfer_values_cache(cache_path=getFileName(False,namesTest[i]), images=np.expand_dims(valX[i],axis=0), model=inceptionModel))
            if i==0:
                valFeatures = transfer_values_val
            else:
                valFeatures = np.vstack(( valFeatures, transfer_values_val ))
        
        feed_dict_val = {processedImages: valXGray,  inceptionFeatureVec:valFeatures}
        predictedClasses = session.run(y_pred_cls, feed_dict_val)
        finalList = finalList + predictedClasses.tolist()
        imageNames = imageNames + namesTest.tolist()
        #print("[%d]: Val:%f" % (batchCount, valAcc))
        batchCount += 1
        if batchCount == numberOfBatches:
            break
    return imageNames,finalList


def predictValidation(inceptionModel, valYielder):
    batchCount = 0
    accuracyList = []
    while(True):
        valX, valXGray, valY, numberOfBatches, namesVal = next(valYielder)
        y_valid_batch = toOneHotEncodingVectorForm(valY.flatten(),0,19)            
        for i in range(valX.shape[0]):
            transfer_values_val = np.array(transfer_values_cache(cache_path=getFileName(False,namesVal[i]), images=np.expand_dims(valX[i],axis=0), model=inceptionModel))
            if i==0:
                valFeatures = transfer_values_val
            else:
                valFeatures = np.vstack(( valFeatures, transfer_values_val ))
        
        feed_dict_val = {processedImages: valXGray,  inceptionFeatureVec:valFeatures, y_true: y_valid_batch}
        valAcc = session.run(accuracy, feed_dict=feed_dict_val)*100
        accuracyList.append(valAcc)
        #print("[%d]: Val:%f" % (batchCount, valAcc))
        batchCount += 1
        if batchCount == numberOfBatches:
            break
    return (sum(accuracyList)*1.00/len(accuracyList))
    
def train(num_iteration, inceptionModel, trainProcessedYielder,valProcessedYielder, valYielder, typeRun):
    total_iterations = 0
    for iteration in range(num_iteration):
        currentResult = predictValidation(inceptionModel,valYielder)
        fileHandler = open(logFile, "a")
        fileHandler.write("======= Accuracy:  ========== {0}\n".format(currentResult))
        print("======= Accuracy: %f ==========" % currentResult)
        fileHandler.close()
        
        batchCount = 0
        while(True):
            trainBatchX,trainXProcessed, trainBatchY,numberOfBatches,namesTrain=next(trainProcessedYielder)
            y_true_batch = toOneHotEncodingVectorForm(trainBatchY.flatten(),0,19)
            
            valX, valXProcessed,valY, _, namesVal = next(valProcessedYielder)
            y_valid_batch = toOneHotEncodingVectorForm(valY.flatten(),0,19)        
            
            for i in range(trainBatchX.shape[0]):
                transfer_values_train = np.array(transfer_values_cache(cache_path=getFileName(True,namesTrain[i]), images=np.expand_dims(trainBatchX[i],axis=0), model=inceptionModel))
                if i==0:
                    trainFeatures = transfer_values_train
                else:
                    trainFeatures = np.vstack( (trainFeatures, transfer_values_train ))

            
            for i in range(valX.shape[0]):
                transfer_values_val = np.array(transfer_values_cache(cache_path=getFileName(False,namesVal[i]), images=np.expand_dims(valX[i],axis=0), model=inceptionModel))
                if i==0:
                    valFeatures = transfer_values_val
                else:
                    valFeatures = np.vstack(( valFeatures, transfer_values_val ))
            
             
            feed_dict_tr = {processedImages: trainXProcessed, inceptionFeatureVec:trainFeatures,  y_true: y_true_batch}
            feed_dict_val = {processedImages: valXProcessed,  inceptionFeatureVec:valFeatures, y_true: y_valid_batch}
            
            session.run([optimizer, cross_entropy], feed_dict=feed_dict_tr)
            trainAcc = session.run(accuracy, feed_dict=feed_dict_tr)*100
            valAcc = session.run(accuracy, feed_dict=feed_dict_val)*100
            fileHandler = open(logFile, "a")
            fileHandler.write("[{0}-{1}]: Train:{2}, Val:{3}\n".format(batchCount, iteration, trainAcc, valAcc))
            fileHandler.close()
            print("[%d-%d]: Train:%f, Val:%f" % (batchCount, iteration, trainAcc, valAcc))
            batchCount += 1
            if batchCount == numberOfBatches:
                break
        saver.save(session, "./checkpoints/iter_"+ typeRun+ str(num_iteration)+ typeOfFile+ time.strftime("%Y%m%d-%H%M%S") +".ckpt")
    total_iterations += num_iteration
    currentResult = predictValidation(inceptionModel, valProcessedYielder)
    fileHandler = open(logFile, "a")
    fileHandler.write("======= Accuracy:  ========== {0}\n".format(currentResult))
    print("======= Accuracy: %f ==========" % currentResult)
    fileHandler.close()
    

writeImageValues()
    


train(3,model, edgeYielder, val_edgeYielder, valYielder, 'edge')
train(3,model, lapYielder, val_lapYielder, valYielder, 'lap')
train(6,model, grayYielder, valYielder, valYielder, 'gray') 


imageNames,predictedValues = predictTest(model, testYielder)
predictedLabels = mapPredictedValuesToLabels(predictedValues,class_index_map_dict)
print("======Done================")

resultFileHandler = open("./result_int_"+typeOfFile +".csv", "w")
resultFileHandler.write("ImageName,ClassId\n")
for testCase in range(len(imageNames)):
    resultFileHandler.write("{0},{1}\n".format(imageNames[testCase], predictedValues[testCase]))
resultFileHandler.close()

resultFileHandler = open("./label_"+typeOfFile+".csv", "w")
for testCase in range(len(imageNames)):
    resultFileHandler.write("{0},{1}\n".format(imageNames[testCase], predictedLabels[testCase]))
resultFileHandler.close()

