import torch
import torch.nn as nn

class Decoder(nn.Module):

    stride = (1)
    padding = (1)
    dilation = (1)
    kernel_size = (3)
    output_padding = (0)

    def __init__(
            self,
            ):

        super(Decoder, self).__init__() # allows you to access nn.Module as a parent class
        # This uses the python 2 version of super. The new way to use super is: super()

        # ENCODE ****************************************************************************************

        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=25, kernel_size=self.kernel_size,
                  stride=self.stride, padding=self.padding,dilation=self.dilation, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)


        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=25, out_channels=25 * 2, kernel_size=self.kernel_size,
                  stride=self.stride, padding=self.padding,dilation=self.dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(25 * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)


        # Layer 3
        self.conv3 = nn.Conv1d(in_channels= (25 * 2), out_channels=(25 * 4), kernel_size = self.kernel_size,
                  stride=self.stride, padding=self.padding,dilation=self.dilation, bias=False)
        self.bn3 = nn.BatchNorm1d(25 * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)


        # Layer 4
        self.conv4 = nn.Conv1d(in_channels = (25 * 4), out_channels = (25*8), kernel_size=self.kernel_size,
                  stride=self.stride, padding=self.padding,dilation=self.dilation, bias=False)
        self.bn4 = nn.BatchNorm1d(25 * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)


        # Layer 5
        self.conv5 = nn.Conv1d(in_channels = (25 * 8), out_channels = (25*16), kernel_size=self.kernel_size,
                  stride=self.stride, padding=self.padding,dilation=self.dilation, bias=False)
        self.bn5 = nn.BatchNorm1d(25 * 16)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        
        # fully connected (linear) layers
        self.fcl1 = nn.Linear(in_features = (10000), out_features=20) # the in-features needs to match the size of 1 input (then it works regardless of batch size)
#         self.reluF1 = nn.LeakyReLU(25*16)

#         self.fcl2 = nn.Linear(in_features = (25*16), out_features=2)
        
    def forward(self, input, epoch): # run input through the layers (perform a forward pass to get an output)

        # run the input layer
        conv1 = self.conv1( input )
        relu1 = self.relu1( conv1 )

        # run output from layer 1 through layer 2
        conv2 = self.conv2( relu1 ) # read data in to layer
        bn2 = self.bn2( conv2 ) # normalize the input before transforming it
        relu2 = self.relu2( bn2 ) # transform the input to outputs

        # run output from layer 2 through layer 3
        conv3 = self.conv3( relu2 )
        bn3 = self.bn3( conv3 )
        relu3 = self.relu3( bn3 )

        # run output from layer 3 through layer 4
        conv4 = self.conv4( relu3 )
        bn4 = self.bn4( conv4 )
        relu4 = self.relu4( bn4 )

        # run output from layer 3 through layer 4
        conv5 = self.conv5( relu4 )
        bn5 = self.bn5( conv5 )
        relu5 = self.relu5( bn5 )  
        
#         # run output through the final layer
        reshaped5 = relu5.reshape(relu5.size(0), -1) # flatten
   
  
  
#         reshaped5 = relu5.view(1, -1)
#         linear6 = self.fcl1(reshaped5) # run through layer
#         relu6 = self.reluF1(linear6)
    
        linear6 = self.fcl1(reshaped5)
        # use a signmoid function to classify the output from layer 6
        return torch.sigmoid( linear6 )
