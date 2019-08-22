import torch
import torch.nn as nn

class Generator(nn.Module):

    e_stride = (1,1)
    e_padding = (1,1)
    e_dilation = (1,1)
    e_kernel_size = (3,3)
    e_output_padding = (0,0)

    d_stride = (1, 1)
    d_padding = (1, 1)
    d_dilation = (1, 1)
    d_kernel_size = (3,3)
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
        self.e_conv3 = nn.Conv2d(in_channels= (25 * 2), out_channels=(25 * 4), kernel_size = self.e_kernel_size,
                  stride=self.e_stride, padding=self.e_padding,dilation=self.e_dilation, bias=False)
        self.e_bn3 = nn.BatchNorm2d(25 * 4)
        self.e_relu3 = nn.LeakyReLU(0.2, inplace=True)


        # Layer 4
        self.e_conv4 = nn.Conv2d(in_channels= (25 * 4), out_channels=25 * 8, kernel_size=self.e_kernel_size,
                  stride=self.e_stride, padding=self.e_padding,dilation=self.e_dilation, bias=False)
        self.e_bn4 = nn.BatchNorm2d(25 * 8)
        self.e_relu4 = nn.LeakyReLU(0.2, inplace=True)
        
        # Layer 5
        self.e_conv5 = nn.Conv2d(in_channels= (25 * 8), out_channels= 25 * 16, kernel_size=self.e_kernel_size,
                  stride=self.e_stride, padding=self.e_padding,dilation=self.e_dilation, bias=False)
        self.e_bn5 = nn.BatchNorm2d(25*16)
        self.e_relu5 = nn.LeakyReLU(0.2, inplace=True)            

        # Layer 6
        self.e_conv6 = nn.Conv2d(in_channels= (25 * 16), out_channels=100, kernel_size=self.e_kernel_size,
                  stride=self.e_stride, padding=self.e_padding,dilation=self.e_dilation, bias=False)
        self.e_bn6 = nn.BatchNorm2d(100)
        self.e_relu6 = nn.LeakyReLU(0.2, inplace=True)          
        
        # DECODE *******************************************************************************************
        # Layer 0
        self.d_conv0 = nn.ConvTranspose2d(in_channels=100, out_channels=25*16, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)
        self.d_bn0 = nn.BatchNorm2d(25*16)
        self.d_relu0 = nn.ReLU(True)
        
        
        # Layer 1
        self.d_conv1 = nn.ConvTranspose2d(in_channels=25*16, out_channels=25*8, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)
        self.d_bn1 = nn.BatchNorm2d(25*8)
        self.d_relu1 = nn.ReLU(True)


        # Layer 2
        self.d_conv2 = nn.ConvTranspose2d(in_channels=25*8, out_channels=25*4, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)
        self.d_bn2 = nn.BatchNorm2d(25*4)
        self.d_relu2= nn.ReLU(True)


        # Layer 3
        self.d_conv3 = nn.ConvTranspose2d(in_channels=25*4, out_channels=25*2, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)
        self.d_bn3 = nn.BatchNorm2d(25*2)
        self.d_relu3 = nn.ReLU(True)


        # Layer 4
        self.d_conv4 = nn.ConvTranspose2d(in_channels=25*2, out_channels=25, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)
        self.d_bn4 = nn.BatchNorm2d(25)
        self.d_relu4 = nn.ReLU(True) 

        # Layer 5
        self.d_conv5 = nn.ConvTranspose2d(in_channels=25, out_channels=1, kernel_size=self.d_kernel_size,
                           stride=self.d_stride, padding=self.d_padding,dilation=self.d_dilation,
                           output_padding=self.d_output_padding, bias=False)

#         # fully connected (linear) layer
#         self.fcl = nn.Linear(in_features = 25*8*1, out_features=25*8*1)


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
        
        # Layer 5
        out12 = self.e_conv5(out11)
        out13 = self.e_bn5(out12)
        out14 = self.e_relu5(out13)

        # Layer 6
        out14z = self.e_conv6(out14)
        out14y = self.e_bn6(out14z)
        out14x = self.e_relu6(out14y)        
        
        # DECODE ************************************

        # Layer 0
        out14a = self.d_conv0(out14x)
        out14b = self.d_bn0(out14a)
        out14 = self.d_relu0(out14b)
        
        
        # Layer 1
        out15 = self.d_conv1(out14)
        out16 = self.d_bn1(out15)
        out17 = self.d_relu1(out16)

        # Layer 2
        out18 = self.d_conv2(out17)
        out19 = self.d_bn2(out18)
        out20 = self.d_relu2(out19)

        # Layer 3
        out21 = self.d_conv3(out20)
        out22 = self.d_bn3(out21)
        out23 = self.d_relu3(out22)

        # Layer 4
        out24 = self.d_conv4(out23)
        out25 = self.d_bn4(out24)
        out26 = self.d_relu4(out25)
        
        # Layer 5
        out27 = self.d_conv5(out26)
        
        out28 = torch.sigmoid(out27) 
        return out28
