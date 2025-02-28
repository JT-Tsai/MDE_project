import torch
import torch.nn as nn

from . import common
from .models import register

class RDB_Conv(nn.Module):
    def __init__(self, in_channels, growRate, kernel_size = 3):
        super().__init__()
        Cin = in_channels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kernel_size, padding = (kernel_size//2), stride = 1),
            nn.ReLU()
        ])
    
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)
    
class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super().__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0+c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0+C*G, G0, 1, padding = 0, stride = 1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x
    
class RDN(nn.Module):
    def __init__(self, args, conv = common.default_conv):
        super().__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kernel_size = args.RDNksize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kernel_size, padding = (kernel_size//2), stride = 1)
        self.SFENet2 = nn.Conv2d(G0, G0, kernel_size, padding = (kernel_size//2), stride = 1)

        # Residual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding = 0, stride = 1),
            nn.Conv2d(G0, G0, kernel_size, padding = (kernel_size//2), stride = 1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    conv(G0, G * r * r, kernel_size),
                    nn.PixelShuffle(r),
                    conv(G, args.n_colors, kernel_size)
                ])
            elif r == 4:
                self.UPNet == nn.Sequential(*[
                    conv(G0, G * 4, kernel_size),
                    nn.PixelShuffle(2),
                    conv(G, G * 4, kernel_size),
                    nn.PixelShuffle(2),
                    conv(G, args.n_colors, kernel_size)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)

@register("rdn")
def make_rdn(G0 = 64, RDNkSize = 3, RDNconfig = 'B', scale = 2, no_upsampling = True):
    args = Namespace()
    args.G0 = G0
    args.RDNksize = RDNkSize
    args.RDNconfig = RDNconfig

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3

    return RDN(args)


if __name__ == "__main__":
    from argparse import Namespace
    from torchinfo import summary

    from .models import make
    
    model_spec = {
        'name': 'rdn',
        'args': {
        #     'G0': 64,
        #     'RDNkSize': 3,
        #     'RDNconfig': 'B',
        #     'scale': 2,
        #     'no_upsampling': True
        },
        'sd': None
    }

    model = make(model_spec)
    print(model)
    summary(model, input_size=(1, 3, 960, 640))
