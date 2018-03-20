
import torch.nn as nn
import torch

oc = 22

class Seg(nn.Module):
    def __init__(self):
        super(Seg, self).__init__()
        self.conv = nn.ModuleList(
                [nn.Conv2d(2048, 32, 1),
                 nn.Conv2d(1024, 32, 1),
                 nn.Conv2d(512, 16, 1),
                 nn.Conv2d(256, 8, 1),
                 nn.Conv2d(64, 4, 1)
                 ])
        
        self.upconv = nn.ModuleList(
                [nn.Conv2d(64, 64, 1),
                                     
                 nn.Sequential(
                         nn.Conv2d(80, 80, 1),
                         nn.UpsamplingBilinear2d(scale_factor=2)),
                 nn.Sequential(
                         nn.Conv2d(88, 88, 1),
                         nn.UpsamplingBilinear2d(scale_factor=2)),
                 nn.Sequential(
                         nn.Conv2d(92, oc, 1),
                         nn.UpsamplingBilinear2d(scale_factor=2))
                 ])
        
    def forward(self, feats):
        feats = feats[::-1]
#        for i in range(5):
#            print(feats[i].size())
#        feats_list = []
        for i in range(5):
            feats[i] = self.conv[i](feats[i])
#        print()
#        for i in range(5):
#            print(feats[i].size())
        
        ff = feats[0]
        i = 1
        for l in self.upconv:
#            print(ff.size())
            ff = torch.cat((ff, feats[i]), 1)
            ff = l(ff)
            i += 1
            
        return ff
            

    