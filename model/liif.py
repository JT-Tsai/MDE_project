import torch
import torch.nn as nn
import torch.nn.functional as F

import edsr
import rdn
from models import make

from utils import make_coord, build

class LIIF(nn.Module):
    def __init__(self, encoder_spec, imnet_spec = None, local_ensembel = True, feat_unfold = True, cell_decode = True):
        super().__init__()
        self.local_ensemble = local_ensembel
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            # concat feature around 3*3 patch
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            # each cell pixel size
            if self.cell_decode:
                imnet_in_dim += 2
                self.imnet = make(imnet_spec, args = {'in_dim': imnet_in_dim})
        else:
            self.imnet = None
    
    def gen_feat(self, input):
        self.feat = self.encoder(input)
        return self.feat
    
    def query_rgb(self, corrd, cell = None):
        feat = self.feat

        if self.imnet is None:
            pass

        if self.feat_unfold:
            pass
            
        if self.local_ensemble:
            pass
        else:
            pass
        