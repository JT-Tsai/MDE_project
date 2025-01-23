import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import register
from .models import make

from utils import make_coord

@register('liif')
class LIIF(nn.Module):
    """
        local_ensemble:
        feat_unfold:
        cell_decode:
    """
    def __init__(self, encoder_spec, imnet_spec = None, local_ensembel = True, feat_unfold = True, cell_decode = True, ):
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

        # focal length estimation
        focal_layer = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]

        out_dim = self.encoder.out_dim
        while out_dim > 16:
            focal_layer.append(nn.Linear(out_dim, out_dim // 2))
            focal_layer.append(nn.ReLU(True))
            out_dim = out_dim // 2

        focal_layer.append(nn.Linear(out_dim, 1))

        self.focal_layers = nn.Sequential(*focal_layer)
    
    def gen_feat(self, input):
        self.feat = self.encoder(input)
        return self.feat
    
    def gen_focal_length(self):
        self.focal_length = self.focal_layers(self.feat)
        return self.focal_length
    
    def query_rgb(self, coord, cell = None):
        feat = self.feat

        if self.imnet is None:
            # flip -> ij coord into xy coord
            # unsqueeze(1) -> insert second dimension (batch_size, 1 (insert_dim), flatten_pixel_num, 2)
            # output -> (batch_size, channels_num, 1, flatten_pixel_num) -> (batch_size, flatten_pixel_num, channels_num)

            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                        mode = 'nearest', align_corners = False)[:, :, 0, :].permute(0, 2, 1)

            return ret
        
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding = 1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
            )
            
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten = False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        # [2 Channel (xy dimension), H, W] -> [1, 2, H, W] -> [n_feat, 2, H, W] 
        
        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                # 
                coord_ = coord.clone()
                # consider around xy coord information
                # mapping into feat shape
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                # clamp_ directly do inplace method
                coord_.clamp_(-1+1e-6, 1-1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode = 'nearest', align_corners = False)[:, :, 0, :].permute(0, 2, 1)
                # (batch_size, flatten_pixel_num, channels_num)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                # (batch_size, flatten_pixel_num, 2 (pixel coord))
                rel_coord = coord - q_coord
                # make coord from [-1, 1] to [H, W]
                rel_coord[:, :, 0] *= feat.shape[-2] # w
                rel_coord[:, :, 1] *= feat.shape[-1] # H

                inp = torch.cat([q_feat, rel_coord], dim = -1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim = -1)

                # batch_size, flatten_pixel_num
                bs, q = coord.shape[:2]
                # pred shape -> [batch_size, flatten_pixel_num, channels_num] -> [batch_size * flatten_pixel_num, channels_num]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim = 0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret
    
    def forward(self, inp, coord, cell = None):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell), self.gen_focal_length()
    

if __name__ == "__main__":
    
    from torchsummary import summary
    import ipdb

    encoder_spec = {
        'name': 'edsr_baseline',
        'args': {
            # 'n_resblock': 32,
            'n_feats': 256,
            # 'res_scale': 0.1,
            # 'scale': 2,
            'no_upsampling': True,
            # 'rgb_range': 1
        },
        'sd': None
    }
    imnet_spec = {
        'name': 'mlp',
        'args': {
            'in_dim': 258,
            'out_dim': 3,
            'hidden_list': [256, 256, 256, 256]
        }
    }
    
    # from .models import make
    # model = make(encoder_spec).cuda()
    # # print(model)
    # summary(model, input_size=(3, 64, 32), batch_size=1)

    model = LIIF(encoder_spec, imnet_spec=imnet_spec, cell_decode=False).cuda()

    print(model)
    # H, W = 35,35
    # coord = make_coord((H, W))
    # input = torch.rand(3, 16, 16)
    # out = model(input, coord)
    # print(out)

    input = torch.rand(1, 3, 4, 4).cuda()
    # batch size = 1
    coord = make_coord((10, 5)).unsqueeze(0).cuda()

    out, focal_length = model(input, coord)
    print(out.shape, focal_length.shape)