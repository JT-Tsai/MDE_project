import os
import time
import shutil
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n
    
    def item(self):
        return self.v
    
class Timer():

    def __init__(self):
        self.v = time.time()
    
    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v
    
def time_text(t):
    if t > 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t > 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)
    
_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename = "log.txt"):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file = f)

def ensure_path(path, remove = True):
    basename = os.path.basename(path.rstrip('/')) # remove trailing slash
    if os.path.exists(path):
        if remove and (basename.startswith('_')
            or input('{} exists, remove? (y/[n]): '.format(path) == 'y')):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def set_save_path(save_path, remove = True):
    ensure_path(save_path, remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer

def compute_num_params(model, text = False):
    tot = int(sum(np.prod(p.shape) for p in model.parameters()))
    if text:
        if tot > 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        if tot > 1e3:
            return '{:.1f}K'.format(tot / 1e3)
        else:
            return tot
        
def make_optimizer(param_list, optimizer_spec, load_sd = False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

def make_coord(shape, ranges = None, flatten = True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim = -1)
    
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """
        coord: Tensor, (pixels_num, 2)
        img: Tensor, (pixels_num, 3_channel_dim)
    """
    coord = make_coord(img.shape[-2:]) # (H, W)
    flatten_rgb = img.view(3, -1).permute(1, 0) # (pixels_num, 3_channel_dim)
    return coord, flatten_rgb

def PSNR(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return math.inf
    else:
        return 20 * math.log10(255 / math.sqrt(mse))
    
def eval_psnr(loader, model, eval_bsize = 5000):
    model.eval()

    psnr = PSNR()
    res = Averager()
    
    pbar = tqdm(loader, desc = 'eval')
    for batch in pbar:
        input = batch['lr_image'].cuda()
        gt = batch['hr_image'].cuda()
        coord = batch['hr_coord'].cuda()
        cell = batch['cell'].cuda()

        with torch.no_grad():
            model.gen_feat(input)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + eval_bsize, n)
                pred = model.query_rgb(coord[:, ql:qr, :], 
                    cell[:, ql:qr, :] if cell is not None else None)
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim = 1)


        h, w = batch['hr_shape']
        shape = [input.shape[0], h, w, 3]
        pred = pred.view(*shape) \
            .permute(0, 3, 1, 2).contiguous()
        gt = gt.view(*shape).permute(0, 3, 1, 2).contiguous()

        val = psnr(pred, gt)
        res.add(val.item(), input.shape[0])

        pbar.set_description('PSNR: {:.4f}'.format(res.item()))

    return res.item()



    


def check_vram():
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    print(mem_used_MB)
