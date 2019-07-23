from Discriminator import *
from Generator import *
from HelperFunctions import *

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


def main():
    transferLearning = False
    fineTuning = False
    cuda = True
    learning_rate = 0.0002
    model_save_interval = 900
    image_save_interval = 900
    update_interval = 3
    log_interval = 100
    gan_curriculum = 10000
    starting_rate = 0.01
    default_rate = 0.5

    epoch_size = 15
    batch_size = 10

    result_path = "gitter_1_numpy_results"
    model_path = "gitter_1_models"
    saved_model_path = "gitter_1_models/16.0"
    saved_gen_A = "model_gen_A-16.0"
    saved_gen_B = "model_gen_B-16.0"
    saved_dis_A = "model_dis_A-16.0"
    saved_dis_B = "model_dis_B-16.0"

    # unload the data files
    data_A, data_B, test_A, test_B = get_data2()
    
    # Initialize Learning Networks
    generator_A = Generator()
    generator_B = Generator()
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    if transferLearning or fineTuning:
        device = None

        saved_gen_A_path = os.path.join(saved_model_path, saved_gen_A)
        saved_gen_B_path = os.path.join(saved_model_path, saved_gen_B)
        saved_dis_A_path = os.path.join(saved_model_path, saved_dis_A)
        saved_dis_B_path = os.path.join(saved_model_path, saved_dis_B)

        if not cuda:
            device = torch.device('cpu')

            gen_A_state_dict = torch.load(saved_gen_A_path, map_location = "cpu")
            gen_B_state_dict = torch.load(saved_gen_B_path, map_location = "cpu")
            dis_A_state_dict = torch.load(saved_dis_A_path, map_location = "cpu")
            dis_B_state_dict = torch.load(saved_dis_B_path, map_location = "cpu")
        else:
            device = torch.device('cuda')

            gen_A_state_dict = torch.load(saved_gen_A_path)
            gen_B_state_dict = torch.load(saved_gen_B_path)
            dis_A_state_dict = torch.load(saved_dis_A_path)
            dis_B_state_dict = torch.load(saved_dis_B_path)

        # obtain the state dictionaries previously trained models



        # load state dictionaries
        generator_A.load_state_dict(gen_A_state_dict)
        generator_B.load_state_dict(gen_B_state_dict)
        discriminator_A.load_state_dict(dis_A_state_dict)
        discriminator_B.load_state_dict(dis_B_state_dict)

        # send dictionaries to device
        generator_A.to(device)
        generator_B.to(device)
        discriminator_A.to(device)
        discriminator_B.to(device)




    # Enable GPUs
    if cuda:
        test_A = test_A.cuda()
        test_B = test_B.cuda()
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    data_size = min(len(data_A), len(data_B))
    n_batches = (data_size // batch_size)

    # Set up different loss functions
    recon_criterion = nn.MSELoss()
    gan_criterion = nn.BCELoss()
    feat_criterion = nn.HingeEmbeddingLoss()

    # Obtain parameters to pass to optimiser
    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    # Setting up gradient descent (optimiser, using the Adam algorithm)
    optim_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5, 0.999),
                           weight_decay=0.00001)  # Default learning_rate is 0.0002
    optim_dis = optim.Adam(dis_params, lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

    iters = 0

    for epoch in range(epoch_size):
        # Shuffle the order of all the data

        data_A, data_B = shuffle_data(data_A, data_B)

        # Progression bar
        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=n_batches, widgets=widgets)
        pbar.start()

        # If the epoch is greater than 5,
        # stop updating the fully connected
        # layers on the generators and discriminators
        if (epoch <= 500) and not (transferLearning) and not (fineTuning):
            for p in generator_A.fcl.parameters():
                p.requires_grad = True
            for p in generator_B.fcl.parameters():
                p.requires_grad = True
            for p in discriminator_A.fcl.parameters():
                p.requires_grad = True
            for p in discriminator_B.fcl.parameters():
                p.requires_grad = True
        else:
            for p in generator_A.fcl.parameters():
                p.requires_grad = False
            for p in generator_B.fcl.parameters():
                p.requires_grad = False
            for p in discriminator_A.fcl.parameters():
                p.requires_grad = False
            for p in discriminator_B.fcl.parameters():
                p.requires_grad = False

        # for each batch
        for i in range(n_batches - 1):

            pbar.update(i)

            # Reset gradients
            generator_B.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            # Get the batches
            A = getBatch(data_A, i, batch_size)
            B = getBatch(data_B, i, batch_size)

            # This returns a batch of dimension batch_size, in_chanels, height, width (30,3,25,8)

            # Enable GPUs
            if cuda:
                A = A.cuda()
                B = B.cuda()

            # PERFORM TRANSLATIONS ON BATCHES **********************************************

            # Convert between domains
            AB = generator_B(A, batch_size)  # generator_B maps from A to B
            BA = generator_A(B, batch_size)

            # Re-convert to original domain
            ABA = generator_A(AB, batch_size)  # Should be back to original images
            BAB = generator_B(BA, batch_size)

            # GENERATE LOSS ***************************************************************

            # Reconstruction Loss: Determine how well original image was reconstructed
            recon_loss_A = recon_criterion(ABA, A)
            recon_loss_B = recon_criterion(BAB, B)

            # Real/Fake GAN Loss (A)
            A_dis_real, A_feats_real = discriminator_A(A, epoch)  # How well does the real A image fit the A domain?
            A_dis_fake, A_feats_fake = discriminator_A(BA,
                                                       epoch)  # How well does the B-generated image fit the A domain

            dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, gan_criterion, cuda)
            fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion, cuda)

            # Real/Fake GAN Loss (B)
            B_dis_real, B_feats_real = discriminator_B(B, epoch)
            B_dis_fake, B_feats_fake = discriminator_B(AB, epoch)

            dis_loss_B, gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, gan_criterion, cuda)
            fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, feat_criterion, cuda)

            # Total Loss

            if iters < gan_curriculum:
                rate = starting_rate
            else:
                rate = default_rate

            gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * (1. - rate) + recon_loss_A * rate
            gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * (1. - rate) + recon_loss_B * rate

            gen_loss = gen_loss_A_total + gen_loss_B_total
            dis_loss = dis_loss_A + dis_loss_B


            # UPDATE EDGES BASED ON LOSSES *****************************************************

            if iters % update_interval == 0: # hold the discriminator constant while we update the GAN
                dis_loss.backward()  
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            if iters % log_interval == 0:
                print()
                print("---------------------")
                print("GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean()))
                print("Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean()))
                print("RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean()))
                print("DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean()))

            # save sample images on the save interval
            if iters % image_save_interval == 0:

                # Get example images
                test_A = getBatch(test_A, 0, batch_size)  # always grab the same images
                test_B = getBatch(test_B, 0, batch_size)

                # run them through the networks
                AB = generator_B(test_A, batch_size)
                BA = generator_A(test_B, batch_size)
                ABA = generator_A(AB, batch_size)
                BAB = generator_B(BA, batch_size)

                # make a place to save the test images
                result_subdir_path = os.path.join(result_path, str(iters / image_save_interval))

                if os.path.exists(result_subdir_path):
                    pass
                else:
                    os.makedirs(result_subdir_path)

                model_subdir_path = os.path.join(model_path, str(iters / image_save_interval))

                if os.path.exists(model_subdir_path):
                    pass
                else:
                    os.makedirs(model_subdir_path)

                # Save the test images
                n_testset = min(test_A.size()[0], test_B.size()[0])
                for im_idx in range(n_testset):

                    A_val = test_A[im_idx][0].cpu().numpy()
                    B_val = test_B[im_idx][0].cpu().numpy()
                    BA_val = BA[im_idx][0].detach().cpu().numpy()
                    ABA_val = ABA[im_idx][0].detach().cpu().numpy()
                    AB_val = AB[im_idx][0].detach().cpu().numpy()
                    BAB_val = BAB[im_idx][0].detach().cpu().numpy()


                    filename_prefix = os.path.join(result_subdir_path, str(im_idx))

                    np.save(filename_prefix + ".A", A_val)
                    np.save(filename_prefix + ".B", B_val)
                    np.save(filename_prefix + ".BA", BA_val)
                    np.save(filename_prefix + ".AB", AB_val)
                    np.save(filename_prefix + ".ABA", ABA_val)
                    np.save(filename_prefix + ".BAB", BAB_val)

            # save models at the save interval
            if iters % model_save_interval == 0:
                torch.save(generator_A.state_dict(),
                           os.path.join(model_subdir_path, 'model_gen_A-' + str(iters / model_save_interval)))
                torch.save(generator_B.state_dict(),
                           os.path.join(model_subdir_path, 'model_gen_B-' + str(iters / model_save_interval)))
                torch.save(discriminator_A.state_dict(),
                           os.path.join(model_subdir_path, 'model_dis_A-' + str(iters / model_save_interval)))
                torch.save(discriminator_B.state_dict(),
                           os.path.join(model_subdir_path, 'model_dis_B-' + str(iters / model_save_interval)))

            iters += 1
