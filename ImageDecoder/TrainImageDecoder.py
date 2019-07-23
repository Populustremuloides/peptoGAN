# from Decoder import *
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
  
    transferLearning = False
    fineTuning = False
    cuda = True
    model_save_interval = 500
    image_save_interval = 500
    update_interval = 85
    log_interval = 100
    
    testingAccuracyList = []
    trainingAccuracyList = []
    trainingLossList = []
    testingLossList = []
#    

    epoch_size = 100
    batch_size = 10

    result_path = "toxicity_classifier_results"
    model_path = "toxicity_classifier"
#     saved_model_path = "toxicity_classifier_models"

    saved_dis_A = "model_dis-94.0"


    # unload the data files
    data, dataLabels, test, testLabels = get_data()
    
    
    # Initialize Learning Network
    decoder = Decoder()

    
    if transferLearning or fineTuning:
        device = None

#         saved_dis_A_path = os.path.join(saved_model_path, saved_dis_A)


        if not cuda:
            device = torch.device('cpu')

            dis_A_state_dict = torch.load("model_dis-94.0", map_location = "cpu")

        else:
            device = torch.device('cuda')

            dis_A_state_dict = torch.load("model_dis-94.0")
            
        # obtain the state dictionary of a previously trained model

        decoder.load_state_dict(dis_A_state_dict, strict = False)
        
        # send dictionary to device

        decoder.to(device)

        
    # Enable GPUs
    if cuda:
        data = data.cuda()
        test = test.cuda()
        decoder = decoder.cuda()

    data_size = len(data)
    n_batches = (data_size // batch_size)

    # Set up loss function
    dis_criterion = nn.BCELoss()

    # Obtain parameters to pass to optimiser
    dis_params = decoder.parameters()

    # Setting up gradient descent (optimiser, using the Adam algorithm)
    optim_dis = optim.Adam(dis_params, lr=0.00001, betas=(0.5, 0.999), weight_decay=0.00001)

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
            decoder.zero_grad()

            # Get the batches
            batch, batchLabels = getBatch(data, dataLabels, i, batch_size) # This returns a batch of dimension batch_size, in_chanels, height, width (30,1,25,8)

            # Enable GPUs
            if cuda:
                batch = batch.cuda()
                batchLabels.cuda()


            # Real/Fake GAN Loss (A)
            trainingClassifications = decoder(batch, epoch)  # How well does the real A image fit the A domain?

            trainingLoss = get_dis_loss(trainingClassifications, batchLabels, dis_criterion, cuda)

            # UPDATE EDGES BASED ON LOSSES *****************************************************
         
            trainingLoss.backward()  
            optim_dis.step()

            if iters % log_interval == 0:

                # Test
                testBatch, testBatchLabels = getBatch(test, testLabels, 0, batch_size)  # always grab the same images
                
                testingClassifications = decoder(testBatch, batch_size)
                testingLoss = get_dis_loss(testingClassifications, testBatchLabels, dis_criterion, cuda)
                testingAccuracy = getAccuracy(testingClassifications, testBatchLabels)               
                
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
              
                torch.save(decoder.state_dict(),
                           os.path.join('model_decoder-' + str(iters / model_save_interval)))

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
    outFile = open("dataDict.pickle", "wb")
    pickle.dump(dictionary, outFile)
    
    df = pd.DataFrame(dictionary)
    print(df)
    df.plot.line()
            
main()
