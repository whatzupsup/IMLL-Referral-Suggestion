import torch
from torch import nn
from torch.nn import functional as F

class EnConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob=0.3, resi=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.resi = resi

        self.layers = nn.Sequential(
            nn.Conv3d(self.in_chans, self.out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.out_chans),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(self.drop_prob),
            nn.Conv3d(self.out_chans, self.out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.out_chans),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(self.drop_prob)
        )

    def forward(self, input):
        return torch.add(self.layers(input), input) if self.resi else self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class ClassificationModel(nn.Module):

    def __init__(self, in_chans, chans, drop_prob):
        super().__init__()

        self.in_chans = in_chans
        self.chans = chans
        self.drop_prob = drop_prob
        
        self.ext_feature = nn.Sequential(nn.Conv3d(self.in_chans, self.chans, kernel_size=3, padding=1), nn.InstanceNorm3d(self.chans), nn.LeakyReLU(0.01, inplace=True), nn.Dropout3d(self.drop_prob))
        ch = self.chans
        
        self.en_level_1 = nn.Sequential(EnConvBlock(ch, ch, drop_prob))
        self.down_1 = nn.Sequential(nn.Conv3d(ch, ch*2, kernel_size=3, padding=1, stride=2), nn.InstanceNorm3d(ch*2), nn.LeakyReLU(0.01, inplace=True))
        ch *= 2
        
        self.en_level_2 = nn.Sequential(EnConvBlock(ch, ch, drop_prob))
        self.down_2 = nn.Sequential(nn.Conv3d(ch, ch*2, kernel_size=3, padding=1, stride=2), nn.InstanceNorm3d(ch*2), nn.LeakyReLU(0.01, inplace=True))
        ch *= 2
        
        self.en_level_3 = nn.Sequential(EnConvBlock(ch, ch, drop_prob))
        self.down_3 = nn.Sequential(nn.Conv3d(ch, ch*2, kernel_size=3, padding=1, stride=2), nn.InstanceNorm3d(ch*2), nn.LeakyReLU(0.01, inplace=True))
        ch *= 2
        
        self.en_level_4 = nn.Sequential(EnConvBlock(ch, ch, drop_prob))
        self.down_4 = nn.Sequential(nn.Conv3d(ch, ch*2, kernel_size=3, padding=1, stride=2), nn.InstanceNorm3d(ch*2), nn.LeakyReLU(0.01, inplace=True))
        ch *= 2
        
        self.en_level_5 = nn.Sequential(EnConvBlock(ch, ch, drop_prob, resi=False), EnConvBlock(ch, ch, drop_prob, resi=False)) #Lowest-Level
        
        self.fc = nn.Sequential(
            nn.Linear(ch, 1)
        )

    def forward(self, input):

        output = input

        output = self.ext_feature(output) 
        
        output = self.en_level_1(output)
        output = self.down_1(output)
        
        output = self.en_level_2(output)
        output = self.down_2(output)
        
        output = self.en_level_3(output)
        output = self.down_3(output)
        
        output = self.en_level_4(output)
        output = self.down_4(output)

        output = torch.add(self.en_level_5(output), output) 
        
        output = F.avg_pool3d(output, output.shape[-3:])
        output = output.contiguous().view(1, -1) 
        output = self.fc(output)
            
        return output
