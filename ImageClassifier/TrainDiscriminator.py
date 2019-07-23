# from Discriminator import *
# from Generator import *
# from HelperFunctions import *

import torch
import torch.nn as nn
import torch.optim as optim

import os
import scipy
import scipy.misc
import numpy as np

from itertools import chain
from progressbar import ETA, Bar, Percentage, ProgressBar

import pickle
import pandas as pd

df = None
dictionary = None

def main():
  
    transferLearning = True
    fineTuning = False
    cuda = True
    model_save_interval = 1000
    image_save_interval = 1000
    update_interval = 85
    log_interval = 100
    
    testingAccuracyList = []
    trainingAccuracyList = []
    trainingLossList = []
    testingLossList = []
#    

    epoch_size = 30
    batch_size = 50

    result_path = "transfer2_toxicity_classifier_results"
    model_path = "transfer2_toxicity_classifier"
#     saved_model_path = "toxicity_classifier_models"

    saved_dis_A = "model_dis-14.0"


    # unload the data files
    data, dataLabels, test, testLabels = get_data()
    
    
    # Initialize Learning Network
    discriminator = Discriminator()

    
    if transferLearning or fineTuning:
        device = None

#         saved_dis_A_path = os.path.join(saved_model_path, saved_dis_A)


        if not cuda:
            device = torch.device('cpu')

            dis_A_state_dict = torch.load("transfer1_model_dis-14.0", map_location = "cpu")

        else:
            device = torch.device('cuda')

            dis_A_state_dict = torch.load("transfer1_model_dis-14.0")
            
        # obtain the state dictionary of a previously trained model

        discriminator.load_state_dict(dis_A_state_dict, strict = False)
        
        # send dictionary to device

        discriminator.to(device)

        
    # Enable GPUs
    if cuda:
        data = data.cuda()
        test = test.cuda()
        discriminator = discriminator.cuda()

    data_size = len(data)
    n_batches = (data_size // batch_size)

    # Set up loss function
    dis_criterion = nn.BCELoss()

    # Obtain parameters to pass to optimiser
    dis_params = discriminator.parameters()

    # Setting up gradient descent (optimiser, using the Adam algorithm)
    optim_dis = optim.Adam(dis_params, lr=0.000005, betas=(0.5, 0.999), weight_decay=0.000007)

    iters = 0

    for epoch in range(epoch_size):
        # Shuffle the order of all the data

        data, dataLabels = shuffle_data(data, dataLabels)

        # Progression bar
        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=n_batches, widgets=widgets)
        pbar.start()
        

        # for each batch
        for i in range(n_batches - 1):

            pbar.update(i)

            # Reset gradients
            discriminator.zero_grad()

            # Get the batches
            batch, batchLabels = getBatch(data, dataLabels, i, batch_size) # This returns a batch of dimension batch_size, in_chanels, height, width (30,1,25,8)

            # Enable GPUs
            if cuda:
                batch = batch.cuda()
                batchLabels.cuda()


            # Real/Fake GAN Loss (A)
            trainingClassifications = discriminator(batch, epoch)  # How well does the real A image fit the A domain?

            trainingLoss = get_dis_loss(trainingClassifications, batchLabels, dis_criterion, cuda)

            # UPDATE EDGES BASED ON LOSSES *****************************************************
         
            trainingLoss.backward()  
            optim_dis.step()

            if iters % log_interval == 0:
               
                
                
                
                # Test
#                 testBatch, testBatchLabels = getBatch(test, testLabels, 0, 5999)  # always grab the same images
                
#                 print("sizes before anything:" + str(len(testBatch)) + " " + str(len(testBatchLabels)))
                
                
#                 if cuda:
#                     testBatch = batch.cuda()
#                     testBatchLabels.cuda()
                    
                if cuda:
                    test = test.cuda()
                    testLabels = testLabels.cuda()
          
                startIndex, stopIndex = getStartStop(testLabels)
                
#                print()
#                print("startIndex: " + str(startIndex))
#                print("stopIndex: " + str(stopIndex))
          
                testingClassifications = discriminator(test[startIndex:stopIndex], 0)
                
#                 print()
                
#                 print("size of testingClassifications:" + str(len(testingClassifications)))
#                 print("size of testBatchLabels:" + str(len(testBatchLabels)))
#                 print("size of testBatch:" + str(len(testBatch)))
                
#                 print()
                
#                 print("size of test:" + str(len(test)))
#                 print("size of testLabels:" + str(len(testLabels)))
#                 print("i:" + str(i))
#                 print("i % 5999:" + str(i % 5999))
                
                
                
  
  
                testingLoss = get_dis_loss(testingClassifications, testLabels[startIndex:stopIndex], dis_criterion, cuda)
                testingAccuracy = getAccuracy(testingClassifications, testLabels[startIndex:stopIndex])               
                
                print()
                print("---------------------")
                print("Training Loss:", as_np(trainingLoss.mean()))
                trainingLossList.append(as_np(trainingLoss.mean()))
                print("Training Accuracy:", getAccuracy(trainingClassifications, batchLabels))
                trainingAccuracyList.append(getAccuracy(trainingClassifications, batchLabels))
                print("Testing Loss:", as_np(testingLoss.mean()))     
                testingLossList.append(as_np(testingLoss.mean()))
                print("Testing Accuracy: ", testingAccuracy)
                testingAccuracyList.append(testingAccuracy)
          
            # save models at the save interval
            if iters % model_save_interval == 0:
              
#                 if os.path.exists(model_subdir_path):
#                     pass
#                 else:
#                     os.makedirs(model_subdir_path)
              
                torch.save(discriminator.state_dict(),
                           os.path.join('transfer2_model_dis-' + str(iters / model_save_interval)))

            iters += 1
    print("assigningDictionary")
    dictionary = {
      "TrainingLoss":trainingLossList,
      "TrainingAccuracy":trainingAccuracyList,
      "TestingLoss":testingLossList,
      "TestingAccuracy":testingAccuracyList
    }
    print(dictionary)
    
    import pickle
    outFile = open("transferDataDict.pickle", "wb")
    pickle.dump(dictionary, outFile)
    
    df = pd.DataFrame(dictionary)
    print(df)
    df.plot.line()
            
main()
