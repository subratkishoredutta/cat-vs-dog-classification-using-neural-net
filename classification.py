# -*- coding: utf-8 -*-
"""
Created on Thu May 28 00:38:16 2020

@author: Asus
"""

import os
import cv2
import numpy as np
from random import shuffle 
from tqdm import tqdm
import tensorflow as tf

train_path =  'D:/KAGGEL/dog_vs_cat/train'
dev_path = 'D:/KAGGEL/dog_vs_cat/training_set'

def create_label(name):
    label = name.split('.')[0]
    
    if label == 'cat':
        return [1., 0.]
    elif label == 'dog':
        return [0., 1.]
    
def create_training_set(size):
    training_data=[]
    for name in tqdm(os.listdir(train_path)):
        label = create_label(name)
        path = os.path.join(train_path,name)
        img = cv2.imread(path,0)
        img = cv2.resize(img,(size,size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    X=[]
    Y=[]
    for i in range(len(training_data)):
        X.append(training_data[i][0])
        Y.append(training_data[i][1])
    return X,Y

def create_dev_set(size):
    dev_data=[]
    for name in tqdm(os.listdir(dev_path)):
        label=create_label(name)
        path = os.path.join(dev_path,name)
        img = cv2.resize(cv2.imread(path,0),(size,size))
        dev_data.append([np.array(img),np.array(label)])
    shuffle(dev_data)
    X=[]
    Y=[]
    for i in range(len(dev_data)):
        X.append(dev_data[i][0])
        Y.append(dev_data[i][1])
    return X,Y

train_X,train_Y = create_training_set(50)
dev_X,dev_Y = create_dev_set(50)

train_X = tf.keras.utils.normalize(train_X,axis=1)
dev_X = tf.keras.utils.normalize(dev_X,axis=1)

train_X_reshaped=np.array(train_X)
train_X_reshaped = train_X_reshaped.reshape(train_X_reshaped.shape[0],train_X_reshaped.shape[1]*train_X_reshaped.shape[1])
#train_X_reshaped = train_X_reshaped.T
dev_X_reshaped=np.array(dev_X)
dev_X_reshaped = dev_X_reshaped.reshape(dev_X_reshaped.shape[0],dev_X_reshaped.shape[1]*dev_X_reshaped.shape[1])

train_Y_reshaped = np.array(train_Y)
dev_Y_reshaped = np.array(dev_Y)
#train_Y_reshaped = train_Y_reshaped.T

ndim = train_X_reshaped.shape[1]

##helper function
## function for creating batches
def create_batch(dataX,dataY,batch_size):
    m=dataX.shape[0]
    nocb = int(m/batch_size)
    remainingSamples = m%batch_size
    batches = []
    for i in range(nocb):
        X=dataX[batch_size*i:batch_size*(i+1)]
        Y=dataY[batch_size*i:batch_size*(i+1)]
        batches.append((X,Y))
    X=dataX[batch_size*(i+1):]
    Y=dataY[batch_size*(i+1):]
    batches.append((X,Y))
    return batches

def accuracy(d,actual):
    count=0
    for i in range(len(d)):
        if d[i] == actual[i] :
            count+=1
    accuracy = count / len(actual)
    print (accuracy)
    return accuracy


def create_placeholders(ndim,class_number):
    
    X = tf.placeholder(tf.float32,[None,ndim])  
    Y = tf.placeholder(tf.float32,[None,class_number])
    return (X,Y)

def initialise_parameters(ndim,class_number):
    W0 = tf.Variable(tf.random_normal([ndim,1024]))
    b0 = tf.Variable(tf.random_normal([1024]))
    
    W1 = tf.Variable(tf.random_normal([1024,1024]))
    b1 = tf.Variable(tf.random_normal([1024]))
    
    W2 = tf.Variable(tf.random_normal([1024,1024]))
    b2 = tf.Variable(tf.random_normal([1024]))
    
    W3 = tf.Variable(tf.random_normal([1024,1024]))
    b3 = tf.Variable(tf.random_normal([1024]))
    
    WO = tf.Variable(tf.random_normal([1024,class_number]))
    bO = tf.Variable(tf.random_normal([class_number]))
    
    parameter={'W0' : W0,
               'W1' : W1,
               'W2' : W2,
               'W3' : W3,
               'WO' : WO,
               'b0' : b0,
               'b1' : b1,
               'b2' : b2,
               'b3' : b3,
               'bO' : bO}
    
    return parameter
    
def forwardpass(X,parameter):
    W0=parameter['W0']
    b0=parameter['b0']
    L0 = tf.add(tf.matmul(X,W0),b0)
    L0 = tf.nn.relu(L0)
    
    W1 = parameter['W1']
    b1 = parameter['b1']
    L1 = tf.add(tf.matmul(L0,W1),b1)
    L1 = tf.nn.relu(L1)
    
    W2 = parameter['W2']
    b2 = parameter['b2']
    L2 = tf.add(tf.matmul(L1,W2),b2)
    L2 = tf.nn.relu(L2)
    
    W3 = parameter['W3']
    b3 = parameter['b3']
    L3 = tf.add(tf.matmul(L2,W3),b3)
    L3 = tf.nn.relu(L3)
    
    WO = parameter['WO']
    bO = parameter['bO']
    
    OL = tf.add(tf.matmul(L3,WO),bO)
    return OL

def costcompute(OL,Y):    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = OL,  labels = Y))
    return cost

def model(Xtrain,Ytrain,LR=0.0001,epoch=1000):
    ndim = Xtrain.shape[1]
    class_number = Ytrain.shape[1]
    
    X,Y = create_placeholders(ndim,class_number)
    
    parameter = initialise_parameters(ndim,class_number)
    
    OL = forwardpass(X,parameter)
    cost = costcompute(OL,Y)
    
    optimiser = tf.train.AdamOptimizer(LR).minimize(cost)
    with tf.Session() as sess:
            
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
        #epoch_loss=0
        #batches = create_batch(train_X_reshaped,train_Y_reshaped,1024)
        #for batch in batches:
        #   epochx,epochy=batch
            _,c=sess.run([optimiser,cost],{X:train_X_reshaped,Y : train_Y_reshaped})
        #    epoch_loss+=c
            if i % 10 == 0 :
                print("loss after",i," number of epoch is:",c)
                
                
        parameters=sess.run(parameter)
        
     
        return parameters
        
    
parameter = model(train_X_reshaped,train_Y_reshaped, LR=1e-4,epoch=7000)


parameter1 = parameter

sess=tf.Session()

X=tf.placeholder(tf.float32,[None,2500])
Y=tf.placeholder(tf.float32,[None,2])
O = forwardpass(X,parameter) 
#g = tf.nn.softmax(Y)  
f=sess.run(O,{X:train_X_reshaped})
pred = np.argmax(f,axis=1)
actual = np.argmax(train_Y_reshaped,axis=1)
train_acc=accuracy(pred,actual)



#g = tf.nn.softmax(Y)  
dev=sess.run(O,{X:dev_X_reshaped})
predDev = np.argmax(dev,axis=1)
actualDev=[]
for i in range(len(dev_Y_reshaped)):
    actualDev.append(np.argmax(dev_Y_reshaped[i]))

actualDev = np.array(actualDev)

dev_acc = accuracy(predDev,actualDev)
            
print("therefore the \n The training accuracy ",train_acc,"\n the dev set accuracy ",dev_acc)



'''   
optimiser = tf.train.AdamOptimizer(0.0001).minimize(cost)


for i in range(10000):
    #epoch_loss=0
    #batches = create_batch(train_X_reshaped,train_Y_reshaped,1024)
    #for batch in batches:
    #   epochx,epochy=batch
    _,c=sess.run([optimiser,cost],{X:train_X_reshaped,Y : train_Y_reshaped})
    #    epoch_loss+=c
    if i % 10 == 0 :
        print("loss after",i," number of epoch is:",c)
correct = tf.equal(tf.argmax(OL,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct,"float"))
    
print ("Train Accuracy:", accuracy.eval({X: train_X_reshaped, Y: train_Y_reshaped}))

d=sess.run(OL,{X:train_X_reshaped})

for i in range(d.shape[0]):
    for j in range(d.shape[1]):
        if d[i][j]>=0.5:
            d[i][j]=1
        elif d[i][j]<0.5:
            d[i][j]=0

accuracy(d,train_Y_reshaped)
    #if i % 10 == 0 :
       # print("loss after",i," number of epoch is:",epoch_loss/1024)
    



'''









