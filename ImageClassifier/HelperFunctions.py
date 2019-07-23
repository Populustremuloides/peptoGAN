from torch.autograd import Variable
import torch
import numpy as np


def get_data():
  
  # get the values
  zero = np.load("antiDomain.npy") # FIXME: put in artificial dataset
  one = np.load("toxicDomain.npy")
  
  # get the labels
  
  zeroLabels = []
  oneLabels = []
  for datum in zero:
    zeroLabels.append([1,0])
  for datum in one:
    oneLabels.append([0,1])
    
  zeroLabels = np.asarray(zeroLabels)
  oneLabels = np.asarray(oneLabels)
  
  
  # concatenate values and labels:
  combinedData = np.concatenate((zero, one))
  combinedLabels = np.concatenate((zeroLabels, oneLabels))
  

  
  
  # randomize them according to the same random indices
  indices = list(range(len(combinedData)))
  np.random.shuffle(indices)
  
  randomizedData = []
  randomizedLabels = []
  for index in indices:
      randomizedData.append(combinedData[index])
      randomizedLabels.append(combinedLabels[index])
    
  
  print(len(randomizedData))
  
  # separate out the test and test labels
  test = randomizedData[0:5999]
  testLabels = randomizedLabels[0:5999]
  
  # put in pytorch format
  randomizedData = np.stack(randomizedData)
  randomizedLabels = np.stack(randomizedLabels)
  
  randomizedData = Variable(torch.FloatTensor(randomizedData))
  randomizedLabels = Variable(torch.FloatTensor(randomizedLabels))
  
  test = np.stack(test)
  testLabels = np.stack(testLabels)
  
  test = Variable(torch.FloatTensor(test))
  testLabels = Variable(torch.FloatTensor(testLabels))
  
  
  return randomizedData, randomizedLabels, test, testLabels
  
def as_np(data):
    return data.cpu().data.numpy()

def shuffle_data(data, labels):
    print(type(data))
    print(type(labels))
  
    if data.shape[0] != labels.shape[0]:
        return None
  
    dataIndex = list(range(data.shape[0]))
    np.random.shuffle( dataIndex )

    shuffledData = []
    shuffledLabels = []
    for index in dataIndex:
        shuffledData.append(as_np(data[index]))
        shuffledLabels.append(as_np(labels[index]))
        
    shuffledData = np.stack(shuffledData)
    shuffledLabels = np.stack(shuffledLabels)
    
    shuffledData = Variable(torch.FloatTensor(shuffledData))
    shuffledLabels = Variable(torch.FloatTensor(shuffledLabels))
    
    return shuffledData, shuffledLabels



def get_dis_loss(output, labels, criterion, cuda):
  

    if cuda:
        output = output.cuda() # not sure we need this, but it can't hurt!
        labels = labels.cuda()

    discriminatorLoss = criterion( output, labels )

    return discriminatorLoss


def getBatch(data, labels, iterations, batchSize):
    
    newBatch = data[iterations * batchSize: (iterations + 1) * batchSize]
    newLabels = labels[iterations * batchSize: (iterations + 1) * batchSize]

    return newBatch, newLabels
  
  
def getAccuracy(classifications, labels):
  
    classifications = as_np(classifications)
    labels = as_np(labels)

    print(classifications.shape)
    print(labels.shape)
    
    numCorrect = 0
    numIncorrect = 0
    for i in range(classifications.shape[0]):
        if classifications[i][0] > classifications[i][1]:
            if labels[i][0] == 1:
                numCorrect = numCorrect + 1
            else:
                numIncorrect = numIncorrect + 1   
        else:
            if labels[i][0] == 1:
                numIncorrect = numIncorrect + 1
            else:
                numCorrect = numCorrect + 1
                
    total = numCorrect + numIncorrect
    
    return (numCorrect / total)
      
      
 import random

def getStartStop(labels):
  
    # This function selects start and stop indices to take a sampel of 100 input images or labels.
    # When used in conjunction with a large testing set of images/labels, this function facilitates
    # testing the accuracy of the discriminator without using large amounts of GPU memeory by selecting
    # a relatively random sample from the testing images and labels.
    
    startIndex = random.randint(0,(len(labels) - 100))
    stopIndex = startIndex + 100
    
    return startIndex, stopIndex
    
