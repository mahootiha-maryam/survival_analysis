import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

class Conv3D_Block(nn.Module):

        def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1):
            super(Conv3D_Block, self).__init__()
    
            self.conv1 = nn.Sequential(
                            nn.Conv3d(inp_feat, out_feat, kernel_size=kernel, 
                                        stride=stride, padding=padding, bias=True),
                            nn.BatchNorm3d(out_feat),
                            nn.ReLU())
    
            self.conv2 = nn.Sequential(
                            nn.Conv3d(out_feat, out_feat, kernel_size=kernel, 
                                        stride=stride, padding=padding, bias=True),
                            nn.BatchNorm3d(out_feat),
                            nn.ReLU())
            
            
        def forward(self, x):
            return self.conv2(self.conv1(x))

class Encoder_FC(nn.Module):
    
    def __init__(self, num_channels=1, feat_channels=[64, 128, 256, 512, 1024], out_features=20):
        
       
        super(Encoder_FC, self).__init__()

        self.pool1 = nn.MaxPool3d((1,2,2))
        self.pool2 = nn.MaxPool3d((1,2,2))
        self.pool3 = nn.MaxPool3d((1,2,2))
        self.pool4 = nn.MaxPool3d((1,2,2))
        
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0])
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1])
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2])
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3])
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4])

        self.relu = nn.ReLU()
        
        self.finalconv1 = nn.Conv3d(feat_channels[4], feat_channels[3] , kernel_size=3, stride=1, padding=0, bias=True)

        self.glob_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, out_features)
        
    def forward(self, x):
        
        x1 = self.conv_blk1(x)
        
        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)
        
        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)
        
        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        seg = self.relu(self.finalconv1(base))

        seg = self.glob_avg_pool(seg)
    
        flatted_seg = torch.flatten(seg, 1)
        final_seg = self.relu(self.fc1(flatted_seg))
        final_seg = self.relu(self.fc2(final_seg))
        final_seg = self.fc3(final_seg)
    
        return final_seg