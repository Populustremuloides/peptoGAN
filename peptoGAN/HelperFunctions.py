from torch.autograd import Variable
import torch
import numpy as np


# Convert 2-dimensional array into 3-d array
# Green and Blue values add location within the image
def addRGB(x):
    threeDList = []
    for image in x:
        g = np.ndarray(image.shape)
        b = np.ndarray(image.shape)
        # add RGb values

        bRow = [0,0.33, 0.66, 0.99, 0.99, 0.66, 0.33, 0]
        gVal = 0
        rowIndex = 0
        for row in image:
            g[rowIndex] = gVal
            b[rowIndex] = bRow

            gVal = gVal + 0.04
            rowIndex = rowIndex + 1
        # combine the three colors into one colored "image"
        threeDImage = np.stack([image, b, g], axis=0)

        threeDList.append(threeDImage)

    threeDList = np.asarray(threeDList)
    return threeDList

def get_data2():
  one = np.load("simple1.npy")
  two = np.load("simple2.npy")
  
  dataA = one
  dataB = two
  
  test_A = Variable(torch.FloatTensor(np.stack(one[0:10])))
  test_B = Variable(torch.FloatTensor(np.stack(two[0:10])))
  
  return dataA, dataB, test_A, test_B
  

def get_data(x, y):

    zero = [] # antitoxic
    one = [] # toxic
    two = [] # neutral

    for imageIndex in range(len(x)):
        # x[imageIndex] = Variable( torch.FloatTensor( x[imageIndex] ), volatile=True )
        if y[imageIndex] == 0:
            zero.append(Variable( torch.FloatTensor( x[imageIndex] )))
        elif y[imageIndex] == 1:
            one.append(Variable( torch.FloatTensor( x[imageIndex] )))
        else:
           two.append(Variable( torch.FloatTensor( x[imageIndex] )))

    data_A = Variable(torch.FloatTensor(np.stack(zero[32:])))
    data_B = Variable(torch.FloatTensor(np.stack(one[32:])))
    test_A = Variable(torch.FloatTensor(np.stack(zero[0:31])))
    test_B = Variable(torch.FloatTensor(np.stack(one[0:31])))
    return data_A, data_B, test_A, test_B



def shuffle_data(da, db):
    a_idx = list(range(len(da)))
    np.random.shuffle( a_idx )

    b_idx = list(range(len(db)))
    np.random.shuffle(b_idx)

    shuffled_da = []
    for index in a_idx:
        shuffled_da.append(da[index])
    shuffled_db = []
    for index in b_idx:
        shuffled_db.append(db[index])
    # shuffled_da = np.array(da)[ np.array(a_idx) ]
    # shuffled_db = np.array(db)[ np.array(b_idx) ]

    shuffled_da = np.stack(shuffled_da)
    shuffled_db = np.stack(shuffled_db)
    
    shuffled_da = Variable(torch.FloatTensor(shuffled_da))
    shuffled_db = Variable(torch.FloatTensor(shuffled_db))
    
    return shuffled_da, shuffled_db



def get_fm_loss(real_feats, fake_feats, criterion, cuda):
    losses = 0
    for real_feat, fake_feat in zip(real_feats[1:], fake_feats[1:]):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        if cuda:
            loss = criterion( l2, Variable( torch.ones( l2.size() ) ).cuda() )
        else:
            loss = criterion( l2, Variable( torch.ones( l2.size() ) ))
        losses += loss

    return losses




def get_gan_loss(dis_real, dis_fake, criterion, cuda):
    print(dis_real.size())
    print(dis_real)
    print(dis_fake.size())
    print(dis_fake)
    labels_dis_real = Variable(torch.ones( [dis_real.size()[0], dis_real.size()[1]] ))
    labels_dis_fake = Variable(torch.zeros([dis_fake.size()[0], dis_fake.size()[1]] ))
    labels_gen = Variable(torch.ones([dis_fake.size()[0], dis_fake.size()[1]]))

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.9 + criterion( dis_fake, labels_dis_fake ) * 0.1
    gen_loss = criterion( dis_fake, labels_gen )

    return dis_loss, gen_loss


def as_np(data):
    return data.cpu().data.numpy()


def getBatch(data, iterations, batchSize):
#     print(data.shape)
    newBatch = data[iterations * batchSize: (iterations + 1) * batchSize]
#     print(newBatch.shape)
    return newBatch
