import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from pycox.models import LogisticHazard


def train_survmodel(labtrans, dl_train, dl_val):
    """
    Make a Neural Network that contains 1 3d cnn with 16 filters 5*5, 1 3d maxpooling
    2*2,  1 3d cnn with 16 filters 5*5, 1 AdaptiveAvgPool3d, 1 FC layer with 16 input
    and 16 output and the last is a FC layer with 16 input and the output is 
    **the time intervals we have**
    For example we can have 20 time intervals
    """

    class Net(nn.Module):
        def __init__(self, out_features):
            super().__init__()
            # numbers in front of conv3d are input channel,number of filters,kernel size,stride
            self.conv1 = nn.Conv3d(1, 16, 5, 1)
            self.max_pool = nn.MaxPool3d(2)
            self.conv2 = nn.Conv3d(16, 16, 5, 1)
            self.glob_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc1 = nn.Linear(16, 16)
            self.fc2 = nn.Linear(16, out_features)
            self.dropout = nn.Dropout3d(0.25)
            self.dropout1 = nn.Dropout(0.25)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.max_pool(x)
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.glob_avg_pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.fc2(x)
            return x

    net = Net(labtrans.out_features)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # gpunet=net.to(device)
    
    model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    model.set_device(torch.device("cuda:0"))
    
    callbacks = [tt.cb.EarlyStopping(patience=5)]
    epochs = 500
    

    verbose = True
    log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_val)

    
    # PATH='/home/mary/Documents/code_surv'
    # torch.save(model, PATH)
    _ = log.plot()
    
    return model
    
    

    """
    make survival analysis model with Nnet-survival 
    Fit the model for training with early stopping
    """

    callbacks = [tt.cb.EarlyStopping(patience=10)]
    epochs = 500
    

    verbose = True
    log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_val)
    
    # PATH='/home/mary/Documents/code_surv'
    # torch.save(model, PATH)
    _ = log.plot()
    
    return model

def train_survmodel_Unet(labtrans, dl_train, dl_val):
    from torch.nn import Module, Sequential 
    from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Linear,AdaptiveAvgPool3d
    from torch.nn import ReLU, Sigmoid
    import torch

    
    

    

    class DoubleConv(torch.nn.Module):
        """
        Helper Class which implements the intermediate Convolutions
        """
        def __init__(self, in_channels, out_channels):
            
            super().__init__()
            self.step = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                            torch.nn.ReLU())
            
        def forward(self, X):
            return self.step(X)
    
        
        


    class UNet3D(Module):
        # __                            __
        #  1|__   ________________   __|1
        #     2|__  ____________  __|2
        #        3|__  ______  __|3
        #           4|__ __ __|4 
        # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity 

        def __init__(self, num_channels=1, feat_channels=[64, 128, 256, 512, 1024], residual='conv'):
            
            # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

            super(UNet3D, self).__init__()
            
            # Encoder downsamplers
            self.pool1 = MaxPool3d((1,2,2))
            self.pool2 = MaxPool3d((1,2,2))
            self.pool3 = MaxPool3d((1,2,2))
            self.pool4 = MaxPool3d((1,2,2))
            
            self.pool5 = MaxPool3d((2,2,2))
            self.pool6 = MaxPool3d((2,2,2))
            
            # Encoder convolutions
            self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
            self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
            self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
            self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
            self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

            # Decoder convolutions
            # self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[3], feat_channels[3], residual=residual)
            # self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], feat_channels[2], residual=residual)
            # self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[1], residual=residual)
            # self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0], residual=residual)

            # # Decoder upsamplers
            # self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
            # self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
            # self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
            # self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

            
            # Activation function
            self.sigmoid = Sigmoid()
            self.relu = ReLU()
            
            
            
            self.finalconv1 = Conv3d(feat_channels[0], 32 , kernel_size=3, stride=1, padding=0, bias=True)
            self.finalconv2 = Conv3d(32, 16 , kernel_size=3, stride=1, padding=0, bias=True)
            self.finalconv3 = Conv3d(16, 16 , kernel_size=3, stride=1, padding=0, bias=True)
            
            self.glob_avg_pool = AdaptiveAvgPool3d((1, 1, 1))
            
            self.fc1 = Linear(16, 16)
            self.fc2 = Linear(16, 10)
            

        def forward(self, x):
            
            # Encoder part
            
            x1 = self.conv_blk1(x)
            
            x_low1 = self.pool1(x1)
            x2 = self.conv_blk2(x_low1)
            
            x_low2 = self.pool2(x2)
            x3 = self.conv_blk3(x_low2)
            
            x_low3 = self.pool3(x3)
            x4 = self.conv_blk4(x_low3)
            
            x_low4 = self.pool4(x4)
            base = self.conv_blk5(x_low4)

            # Decoder part
            
            d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
            d_high4 = self.dec_conv_blk4(d4)

            d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
            d_high3 = self.dec_conv_blk3(d3)

            d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
            d_high2 = self.dec_conv_blk2(d2)

            d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
            d_high1 = self.dec_conv_blk1(d1)
            
            seg = self.sigmoid(self.finalconv1(d_high1))
            seg = self.pool5(seg)
            seg = self.sigmoid(self.finalconv2(seg))
            seg = self.pool6(seg)
            seg = self.sigmoid(self.finalconv3(seg))
            
            seg = self.glob_avg_pool(seg)
            
            flatted_seg = torch.flatten(seg, 1)
            final_seg = self.relu(self.fc1(flatted_seg))
            final_seg = self.fc2(final_seg)
            
            

            return final_seg

        
    class Conv3D_Block(Module):
            
        def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):
        
            super(Conv3D_Block, self).__init__()

            self.conv1 = Sequential(
                            Conv3d(inp_feat, out_feat, kernel_size=kernel, 
                                        stride=stride, padding=padding, bias=True),
                            BatchNorm3d(out_feat),
                            ReLU())

            self.conv2 = Sequential(
                            Conv3d(out_feat, out_feat, kernel_size=kernel, 
                                        stride=stride, padding=padding, bias=True),
                            BatchNorm3d(out_feat),
                            ReLU())
            
            self.residual = residual

            if self.residual is not None:
                self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

        def forward(self, x):
            
            res = x

            if not self.residual:
                return self.conv2(self.conv1(x))
            else:
                return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


    class Deconv3D_Block(Module):
        
        def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
            
            super(Deconv3D_Block, self).__init__()
            
            self.deconv = Sequential(
                            ConvTranspose3d(inp_feat, out_feat, kernel_size=(1,kernel,kernel), 
                                        stride=(1,stride,stride), padding=(0, padding, padding), output_padding=0, bias=True),
                            ReLU())
        
        def forward(self, x):
            
            return self.deconv(x)
        
    
    class UNet(torch.nn.Module):
        """
        This class implements a UNet for the Segmentation
        We use 3 down- and 3 UpConvolutions and two Convolutions in each step
        """
    
        def __init__(self):
            """Sets up the U-Net Structure
            """
            super().__init__()
            
            
            ############# DOWN #####################
            self.layer1 = DoubleConv(1, 32)
            self.layer2 = DoubleConv(32, 64)
            self.layer3 = DoubleConv(64, 128)
            self.layer4 = DoubleConv(128, 256)
    
            #########################################
    
            ############## UP #######################
            self.layer5 = DoubleConv(256 + 128, 128)
            self.layer6 = DoubleConv(128+64, 64)
            self.layer7 = DoubleConv(64+32, 32)
            self.layer8 = torch.nn.Conv3d(32, 3, 1)  # Output: 3 values -> background, liver, tumor
            #########################################
    
            self.maxpool = torch.nn.MaxPool3d(2)
    
        def forward(self, x):
            
            ####### DownConv 1#########
            x1 = self.layer1(x)
            x1m = self.maxpool(x1)
            ###########################
            
            ####### DownConv 2#########        
            x2 = self.layer2(x1m)
            x2m = self.maxpool(x2)
            ###########################
    
            ####### DownConv 3#########        
            x3 = self.layer3(x2m)
            x3m = self.maxpool(x3)
            ###########################
            
            ##### Intermediate Layer ## 
            x4 = self.layer4(x3m)
            ###########################
    
            ####### UpCONV 1#########        
            x5 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x4)  # Upsample with a factor of 2
            x5 = torch.cat([x5, x3], dim=1)  # Skip-Connection
            x5 = self.layer5(x5)
            ###########################
    
            ####### UpCONV 2#########        
            x6 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x5)        
            x6 = torch.cat([x6, x2], dim=1)  # Skip-Connection    
            x6 = self.layer6(x6)
            ###########################
            
            ####### UpCONV 3#########        
            x7 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x6)
            x7 = torch.cat([x7, x1], dim=1)       
            x7 = self.layer7(x7)
            ###########################
            
            ####### Predicted segmentation#########        
            ret = self.layer8(x7)
            return ret    
        
    net = UNet()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpunet=net.to(device)
    
    model = LogisticHazard(gpunet, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
    
    
    

    """
    make survival analysis model with Nnet-survival 
    Fit the model for training with early stopping
    """

    callbacks = [tt.cb.EarlyStopping(patience=10)]
    epochs = 500
    

    verbose = True
    log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_val)
    
    # PATH='/home/mary/Documents/code_surv'
    # torch.save(model, PATH)
    _ = log.plot()
    
    
    
    
    
def train_survmodel_extradata(labtrans, dl_train, dl_val):
    import pandas as pd
    df = pd.read_csv ('/home/mary/Downloads/COMET_Clinical_data.csv')
    
    class Net(nn.Module):
        def __init__(self, out_features):
            super().__init__()
            # numbers in front of conv3d are input channel,number of filters,kernel size,stride
            self.conv1 = nn.Conv3d(1, 16, 5, 1)
            self.max_pool = nn.MaxPool3d(2)
            self.conv2 = nn.Conv3d(16, 16, 5, 1)
            self.glob_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc1 = nn.Linear(16, 16)
            
            
            self.fc2 = nn.Linear(16+4, out_features)
            self.dropout = nn.Dropout3d(0.25)
            self.dropout1 = nn.Dropout(0.25)
            
        def forward(self, image,data):

            
            
            x1 = F.relu(self.conv1(image))
            x1 = self.max_pool(x1)
            x1 = self.dropout(x1)
            x1 = F.relu(self.conv2(x1))
            x1 = self.glob_avg_pool(x1)
            x1 = torch.flatten(x1, 1)
            
            x2 = data
            
            x = torch.cat((x1, x2), dim=1)
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.fc2(x)
            return x

    net = Net(labtrans.out_features)
    
    image = torch.randn(1, 1, 128, 128, 128)
    data = torch.randn(1, 10)
    fnet = net(image,data)
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #gpunet=net.to(device)
    
    model = LogisticHazard(fnet, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
    
    
    

    """
    make survival analysis model with Nnet-survival 
    Fit the model for training with early stopping
    """

    callbacks = [tt.cb.EarlyStopping(patience=10)]
    epochs = 500
    

    verbose = True
    log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_val)
    
    # PATH='/home/mary/Documents/code_surv'
    # torch.save(model, PATH)
    _ = log.plot()
    
    return model

def train_new_based_on_Unet(labtrans, dl_train, dl_val):
    from torch.nn import Module, Sequential 
    from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Linear,AdaptiveAvgPool3d
    from torch.nn import ReLU, Sigmoid
    import torch
    
    class UNet3D_Encoder_FC(Module):
        # __                            __
        #  1|__   ________________   __|1
        #     2|__  ____________  __|2
        #        3|__  ______  __|3
        #           4|__ __ __|4 
        # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity 

        def __init__(self, num_channels=1, feat_channels=[64, 128, 256, 512, 1024], residual='conv'):
            
            # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

            super(UNet3D_Encoder_FC, self).__init__()
            
            # Encoder downsamplers
            self.pool1 = MaxPool3d((1,2,2))
            self.pool2 = MaxPool3d((1,2,2))
            self.pool3 = MaxPool3d((1,2,2))
            self.pool4 = MaxPool3d((1,2,2))
            
            self.pool5 = MaxPool3d((2,2,2))
            self.pool6 = MaxPool3d((2,2,2))
            
            # Encoder convolutions
            self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
            self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
            self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
            self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
            self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

            
            
            
            # Activation function
            self.sigmoid = Sigmoid()
            self.relu = ReLU()
            
            
            
            self.finalconv1 = Conv3d(feat_channels[4], feat_channels[3] , kernel_size=3, stride=1, padding=0, bias=True)
            # self.finalconv2 = Conv3d(32, 16 , kernel_size=3, stride=1, padding=0, bias=True)
            # self.finalconv3 = Conv3d(16, 16 , kernel_size=3, stride=1, padding=0, bias=True)
            
            self.glob_avg_pool = AdaptiveAvgPool3d((1, 1, 1))
            
            self.fc1 = Linear(512, 256)
            self.fc2 = Linear(256, 128)
            self.fc3= Linear(128, 10)
            

        def forward(self, x):
            
            # Encoder part
            
            x1 = self.conv_blk1(x)
            
            x_low1 = self.pool1(x1)
            x2 = self.conv_blk2(x_low1)
            
            x_low2 = self.pool2(x2)
            x3 = self.conv_blk3(x_low2)
            
            x_low3 = self.pool3(x3)
            x4 = self.conv_blk4(x_low3)
            
            x_low4 = self.pool4(x4)
            base = self.conv_blk5(x_low4)

            # Decoder part
            
            # d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
            # d_high4 = self.dec_conv_blk4(d4)

            # d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
            # d_high3 = self.dec_conv_blk3(d3)

            # d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
            # d_high2 = self.dec_conv_blk2(d2)

            # d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
            # d_high1 = self.dec_conv_blk1(d1)
            
            seg = self.sigmoid(self.finalconv1(base))
            seg = self.pool5(seg)
            # seg = self.sigmoid(self.finalconv2(seg))
            # seg = self.pool6(seg)
            # seg = self.sigmoid(self.finalconv3(seg))
            
            seg = self.glob_avg_pool(seg)
            
            flatted_seg = torch.flatten(seg, 1)
            final_seg = self.relu(self.fc1(flatted_seg))
            final_seg = self.relu(self.fc2(final_seg))
            final_seg = self.fc3(final_seg)
            
            

            return final_seg

        
    class Conv3D_Block(Module):
            
        def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):
        
            super(Conv3D_Block, self).__init__()

            self.conv1 = Sequential(
                            Conv3d(inp_feat, out_feat, kernel_size=kernel, 
                                        stride=stride, padding=padding, bias=True),
                            BatchNorm3d(out_feat),
                            ReLU())

            self.conv2 = Sequential(
                            Conv3d(out_feat, out_feat, kernel_size=kernel, 
                                        stride=stride, padding=padding, bias=True),
                            BatchNorm3d(out_feat),
                            ReLU())
            
            self.residual = residual

            if self.residual is not None:
                self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

        def forward(self, x):
            
            res = x

            if not self.residual:
                return self.conv2(self.conv1(x))
            else:
                return self.conv2(self.conv1(x)) + self.residual_upsampler(res)
            
    net = UNet3D_Encoder_FC()
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # gpunet=net.to(device)
    
    model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    model.set_device(torch.device("cuda:0"))
    
    callbacks = [tt.cb.EarlyStopping(patience=5)]
    epochs = 500
    

    verbose = True
    log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_val)

    
    # PATH='/home/mary/Documents/code_surv'
    # torch.save(model, PATH)
    _ = log.plot()
    
    return model