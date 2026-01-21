import torch
import torch.nn as nn

from DiffNet import  FeatureExtractionAndDifferenceEncoder

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.leakly_Relu = nn.LeakyReLU()
    def forward(self, feature_map):
        x = self.conv(feature_map)
        x = self.leakly_Relu(x)
        return x

class DetectionNet(nn.Module):
    def __init__(self):
        super(DetectionNet, self).__init__()

        # Encoder part
        self.encoder = FeatureExtractionAndDifferenceEncoder()

        # Decoder part
        self.high_level_feature = nn.Sequential(Conv(256, 128))
        self.mid_level_feature = nn.Sequential(Conv(160, 64))
        self.low_level_feature = nn.Sequential(Conv(80, 48))

        self.output_layer = nn.Conv2d(3, 3, (1, 1),bias=True)

        self.upsample1 = nn.PixelShuffle(2)
        self.upsample2 = nn.PixelShuffle(2)
        self.upsample3 = nn.PixelShuffle(4)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, images): 
        diff1, diff2, diff3 = self.encoder(images)
    
        x = self.high_level_feature(diff3)
        x = self.upsample1(x)    
        x = torch.cat([x, diff2], dim=1)  
        x = self.mid_level_feature(x)     
        x = self.upsample2(x)
        x = torch.cat([x, diff1], dim=1)
        x = self.low_level_feature(x)
        x = self.upsample3(x)
        heatmaps = self.output_layer(x)
        heatmaps = self.sigmoid(heatmaps)
        return heatmaps
        

