import torch
import torch.nn as nn

class Generator(nn.Module):

    e_stride = (1,1)
    e_padding = (1,1)
    e_dilation = (2,1)
    e_kernel_size = (4,4)
    e_output_padding = (0,0)

    d_stride = (1, 1)
    d_padding = (1, 1)
    d_dilation = (2, 1)
    d_kernel_size = (4,4)
    d_output_padding = (0, 0)

    def __init__(
            self,

            extra_layers=False
            ):

        # allows us to inherit from nn.Module as a parent class
        super(Generator, self).__init__()

        # ENCODE ****************************************************************************************

        # Layer 1
        self.e_conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=self.e_kernel_size,
                  stride=self.e_stride, padding=self.e_padding,dilation=self.e_dilation, bias=False)
        self.e_relu1 = nn.LeakyReLU(0.2, inplace=True)


        # Layer 2
        self.e_conv2 = nn.Conv2d(in_channels=25, out_channels=25 * 2, kernel_size=self.e_kernel_size,
                  stride=self.e_stride, padding=self.e_padding,dilation=self.e_dilation, bias=False)
        self.e_bn2 = nn.BatchNorm2d(25 * 2)
        self.e_relu2 = nn.LeakyReLU(0.2, inplace=True)


        # Layer 3
        self.e_conv3 = nn.Conv2d(in_channels= (25 * 2), out_channels=(25 * 3), kernel_size = self.e_kernel_size,
                  stride=self.e_stride, padding=self.e_padding,dilation=self.e_dilation, bias=False)
        self.e_bn3 = nn.BatchNorm2d(25 * 3)
        self.e_relu3 = nn.LeakyReLU(0.2, inplace=True)


        # Layer 4
        self.e_conv4 = nn.Conv2d(in_channels= (25 * 3), out_channels=25 * 4, kernel_size=self.e_kernel_size,
                  stride=self.e_stride, padding=self.e_padding,dilation=self.e_dilation, bias=False)
        self.e_bn4 = nn.BatchNorm2d(25 * 4)
        self.e_relu4 = nn.LeakyReLU(0.2, inplace=True)

        # DECODE *******************************************************************************************

        # Layer 1
        self.d_conv1 = nn.ConvTranspose2d(in_channels=25*4, out_channels=25*3, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)
        self.d_bn1 = nn.BatchNorm2d(25*3)
        self.d_relu1 = nn.ReLU(True)


        # Layer 2
        self.d_conv2 = nn.ConvTranspose2d(in_channels=25*3, out_channels=25*2, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)
        self.d_bn2 = nn.BatchNorm2d(25*2)
        self.d_relu2= nn.ReLU(True)


        # Layer 3
        self.d_conv3 = nn.ConvTranspose2d(in_channels=25*2, out_channels=25, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)
        self.d_bn3 = nn.BatchNorm2d(25*1)
        self.d_relu3 = nn.ReLU(True)


        # Layer 4
        self.d_conv4 = nn.ConvTranspose2d(in_channels=25, out_channels=1, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)


        # fully connected (linear) layer
        self.fcl = nn.Linear(in_features = 25*8*1, out_features=25*8*1)


    def forward(self, input, batch_size):
        # ENCODE ***********************************

        # Layer 1
        out1 = self.e_conv1(input)
        out2 = self.e_relu1(out1)

        # Layer 2
        out3 = self.e_conv2(out2)
        out4 = self.e_bn2(out3)
        out5 = self.e_relu2(out4)

        # Layer 3
        out6 = self.e_conv3(out5)
        out7 = self.e_bn3(out6)
        out8 = self.e_relu3(out7)

        # Layer 4
        out9 = self.e_conv4(out8)
        out10 = self.e_bn4(out9)
        out11 = self.e_relu4(out10)

        # DECODE ************************************

        # Layer 1
        out12 = self.d_conv1(out11)
        out13 = self.d_bn1(out12)
        out14 = self.d_relu1(out13)

        # Layer 2
        out15 = self.d_conv2(out14)
        out16 = self.d_bn2(out15)
        out17 = self.d_relu2(out16)

        # Layer 3
        out18 = self.d_conv3(out17)
        out19 = self.d_bn3(out18)
        out20 = self.d_relu3(out19)

        # Layer 4
        out21 = self.d_conv4(out20)

        # Fully Connected Layer ********************
        # This layer is here to allow for feature selection on this autoen/decoder
        out22 = out21.reshape(out21.size(0), -1)
        out23 = self.fcl(out22)
        # reshape it back into an image

        out24 = torch.reshape(out23, (batch_size,1,25,8))
        out25 = torch.sigmoid(out24) # not sure why, but DiscoGAN did this :)
#         return out25
        return out25
