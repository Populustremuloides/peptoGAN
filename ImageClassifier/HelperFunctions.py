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
    
  
 # print(len(randomizedData))
  
  # separate out the test, feature, and train sets
  test = randomizedData[0:5999]
  testLabels = randomizedLabels[0:5999]
  
  feature = randomizedData[6000:11999]
  featureLabels = randomizedLabels[6000:11999]
  
  train = randomizedData[12000:]
  trainLabels = randomizedLabels[12000:]
  
  
  # put in pytorch format
  
  train = np.stack(train)
  trainLabels = np.stack(trainLabels)
  

  test = np.stack(test)
  testLabels = np.stack(testLabels)
  
  feature = np.stack(feature)
  featureLabels = np.stack(featureLabels)  
  

   
  return train, trainLabels, test, testLabels, feature, featureLabels
  
def as_np(data):
    return data.cpu().data.numpy()

def shuffle_data(data, labels):
    #print(type(data))
    #print(type(labels))
  
    if data.shape[0] != labels.shape[0]:
        return None
  
    dataIndex = list(range(data.shape[0]))
    np.random.shuffle( dataIndex )

    shuffledData = []
    shuffledLabels = []
    for index in dataIndex:
        shuffledData.append(as_np(data[index]))
        shuffledLabels.append(as_np(labels[index]))

#     shuffledData = torch.Tensor.cpu(shuffledData)
#     shuffledLabels = ensor.cpu(shuffledLabels)
        
    shuffledData = np.stack(shuffledData)
    shuffledLabels = np.stack(shuffledLabels)
    
    shuffledData = Variable(torch.FloatTensor(shuffledData))
    shuffledLabels = Variable(torch.FloatTensor(shuffledLabels))
    
    return shuffledData, shuffledLabels



def get_dis_loss(output, labels, criterion, cuda):
    
#     print()
#     print("in get_dis_loss")
#     print("len output: " + str(output.shape))
#     print("len labels:" + str(labels.shape))
    
#     print(output[0])
#     print(labels[0])

    if cuda:
        output = output.cuda() # not sure we need this, but it can't hurt!
        labels = labels.cuda()

    discriminatorLoss = criterion( output, labels )

    return discriminatorLoss





def getBatch(data, labels, iterations, batchSize):
  
  
#     if len(data) < 30000:
        #print("IN getBatch")
        #print("start:")
        #print(str(iterations * batchSize))
        #print("stop")
        #print(str((iterations + 1) * batchSize))

    
    
    newBatch = data[iterations * batchSize: (iterations + 1) * batchSize]
    newLabels = labels[iterations * batchSize: (iterations + 1) * batchSize]

    return newBatch, newLabels
  
  
def getAccuracy(classifications, labels):
  
    classifications = as_np(classifications)
    labels = as_np(labels)

    #print(classifications.shape)
    #print(labels.shape)
    
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
      
      
  
