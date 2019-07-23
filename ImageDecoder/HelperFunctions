import csv
from torch.autograd import Variable
import torch
import numpy as np
import random


def getRefDict():
    referenceDict = {}
    aminoAcidMatrix = "aaMatrixClean.csv"
    with open(aminoAcidMatrix) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            # make new list
            aminoAcid = ""

            # attributes = np.ndarray(shape = (0,), dtype = float)
            attributes = []
            i = 0
            for value in row:
                if i == 0:
                    aminoAcid = value # save which amino acid this is for
                else:
                    # print(value.replace("\n",""))
                    attributes.append(float(value))
                i = i + 1
            referenceDict[aminoAcid] = np.array(attributes)

    return referenceDict
    
    
    

def makeTrainingData(numSamples):
    numTrainingData = numSamples
    referenceDict = getRefDict()
    
    for key in referenceDict.keys():
        newList = []
        for element in referenceDict[key]:
            newList.append(float(element))
        referenceDict[key] = newList
        
    aminos = ["G","A","P","V","L","I","M","F","Y","W","S","T","C","N","Q","K","H","R","D","E"]
    trainingData = []
    trainingLables = []

    for i in range(numTrainingData):
        index = random.randint(0,19)

        aminoAcid = aminos[index]
        row = [referenceDict[aminoAcid]]
        trainingData.append(row)
        
        newLabel = np.zeros((1,20), dtype=float)
        newLabel[:,index] = 1
        trainingLables.append(newLabel[0])

    trainingData = np.stack(trainingData)
      
    return trainingData, trainingLables
  
  
  
 def getMaxIndex(numbers):
    maxIndex = 0
    i = 0
    for number in numbers:
        if number >= numbers[maxIndex]:
            maxIndex = i
        i = i + 1
    return maxIndex
  
  
  
def getAminoAcid(numbers):
    if len(numbers) != 20:
        return "%"
    else:
        aminos = ["G","A","P","V","L","I","M","F","Y","W","S","T","C","N","Q","K","H","R","D","E"]
        maxIndex = getMaxIndex(numbers)
        
        return aminos[maxIndex]
  
  



def get_data():
  
  
  trainingData, trainingLabels = makeTrainingData(30000)
  testingData, testingLabels = makeTrainingData(30000)
  
  trainingData = Variable(torch.FloatTensor(trainingData))
  trainingLabels = Variable(torch.FloatTensor(trainingLabels))
  testingData = Variable(torch.FloatTensor(testingData))
  testingLabels = Variable(torch.FloatTensor(testingLabels))
 

  
  
  return trainingData, trainingLabels, testingData, testingLabels
  
def as_np(data):
    return data.cpu().data.numpy()

def shuffle_data(data, labels):
  
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
        output = output.cuda() 
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
    
    numCorrect = 0
    numIncorrect = 0
    # for each classification in each batch
    for i in range(classifications.shape[0]):
        maxIndexClassification = getMaxIndex(classifications[i])
        maxIndexLabels = getMaxIndex(labels[i])
        
        if maxIndexClassification == maxIndexLabels:
            numCorrect = numCorrect + 1
        else:
            numIncorrect = numIncorrect + 1
                
    total = numCorrect + numIncorrect
    
    return (numCorrect / total)
  
