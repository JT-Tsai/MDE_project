import common
import torch
import torch.nn as nn

from models import register

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

class EDSR(nn.Module):
    def __init__(self, args, conv = common.default_conv):
        super().__init__()
        self.args = args

        n_resblock = args.n_resblock
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        url_name = 'r{}f{}x{}'.format(n_resblock, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign = 1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act = act, res_scale = args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                common.Upsampler(conv, scale, n_feats, act = False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
 
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict = True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1: # tail part maybe not match to the external model shape
                        raise RuntimeError('While copying the parameter named {},'
                                           'whose dimensions in the model are {} and'
                                           'whose dimensions in the ckeckpoint are {},'
                                           .format(name, own_state[name].size(), state_dict[name].size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))    

@register("edsr_baseline")
def make_edsr_baseline(n_resblock = 16, n_feats = 64, res_scale = 1, scale = 2, no_upsampling = False, rgb_range = 1):
    args = Namespace()
    args.n_resblock = n_resblock
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range
    args.n_colors = 3

    return EDSR(args)

@register("edsr")
def make_edsr(n_resblock = 32, n_feats = 256, res_scale = 0.1, scale = 2, no_upsampling = False, rgb_range = 1):
    args = Namespace()
    args.n_resblock = n_resblock
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range
    args.n_colors = 3

    return EDSR(args)

if __name__ == "__main__":
    from argparse import Namespace
    from torchinfo import summary
    from models import make

    model_spec = {
        'name': 'edsr',
        'args': {
            'n_resblock': 32,
            'n_feats': 256,
            'res_scale': 0.1,
            'scale': 2,
            'no_upsampling': False,
            'rgb_range': 1
        },
        'sd': None
    }

    model = make(model_spec)
    print(model)

    summary(model, input_size=(1, 3, 480, 320))

    input_tensor = torch.randn(1, 3, 480, 320).cuda()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)

