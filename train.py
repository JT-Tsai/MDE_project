import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import MultiStepLR
from torchinfo import summary

import datasets
import models
import utils

from PSNR import PSNR
from SSIM import SSIM

import ipdb

def make_data_loader(spec):
    if spec is None:
        return None
    
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args = {'dataset': dataset})

    test_size = int(len(dataset) * spec['test_ratio'])
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - test_size, test_size])

    log('{} dataset: size = {}'.format('train', len(train_dataset)))
    log('{} dataset: size = {}'.format('val', len(val_dataset)))

    # for k, v in dataset[0].items():
    #     log('  {}: shape={}'.format(k, tuple(v.shape)))

    train_loader = DataLoader(train_dataset, batch_size=spec['batch_size'], 
                        shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=spec['batch_size'], 
                        shuffle = False)
    
    return train_loader, val_loader

def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd = True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd = True
        )
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        
        return model, optimizer, epoch_start, lr_scheduler
    
def train(train_loader, model, optimizer, bsize):
    L1_loss = nn.L1Loss()
    # L2_loss = nn.MSELoss()
    model.train()
    
    pbar = tqdm(train_loader, desc = 'train')
    for batch in pbar:
        input = batch['lr_image'].cuda()
        gt = batch['hr_image'].cuda()
        coord = batch['hr_coord'].cuda()
        cell = batch['cell'].cuda()
        
        model.gen_feat(input)
        n = coord.shape[1]
        ql = 0
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :] if cell is not None else None)
            # ipdb.set_trace()
            loss = L1_loss(pred, gt[:, ql:qr, :])

            pbar.set_description('loss: {:.4f}'.format(loss.item()))
            writer.add_scalars('loss', {'train': loss}, optimizer.step_num)
            
            optimizer.zero_grad()
            loss.backward()
            # ipdb.set_trace()
            optimizer.step()
            
            ql = qr
            pred = None; loss = None

def eval(loader, model, epoch, eval_bsize = 5000):
    model.eval()
    psnr_avg = utils.Averager()
    ssim_avg = utils.Averager()
    PSNR_Metric = PSNR()
    SSIM_Metric = SSIM()

    pbar = tqdm(loader, leave = False, desc = 'eval')
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

        # ipdb.set_trace()
        h, w = batch['hr_shape']
        shape = [input.shape[0], h, w, 3]
        pred = pred.view(*shape) \
            .permute(0, 3, 1, 2).contiguous()
        gt = gt.view(*shape).permute(0, 3, 1, 2).contiguous()

        psnr_val = PSNR_Metric(pred, gt)
        ssim_val = SSIM_Metric(pred, gt)
        # ipdb.set_trace()
        psnr_avg.add(psnr_val.item(), input.shape[0])
        ssim_avg.add(ssim_val.item(), input.shape[0])
        
        # cat with gt for visualization
        # Add enumerate index to track batch order
        ipdb.set_trace()
        utils.save_img(pred, gt, epoch=epoch, idx=pbar.n)

        pbar.set_description('PSNR: {:.4f}, SSIM: {:.4f}'
                             .format(psnr_avg.item(), ssim_avg.item()))

    return psnr_avg.item(), ssim_avg.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    train_loader, val_loader = make_data_loader(config.get('dataset'))

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    # ipdb.set_trace()

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_psnr = -1e18
    max_ssim = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max+1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train(train_loader, model, optimizer, bsize = config['train_bsize'])

        if lr_scheduler is not None:
            lr_scheduler.step()

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
        
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            psnr, ssim = eval(val_loader, model, epoch = epoch, eval_bsize = config['eval_bsize'])

            log_info.append('val: psnr={:.4f}'.format(psnr))
            log_info.append('val: ssim={:.4f}'.format(ssim))
            writer.add_scalars('psnr', {'val': psnr}, epoch)
            writer.add_scalars('ssim', {'val': ssim}, epoch)

            if psnr > max_psnr and ssim > max_ssim:
                max_psnr = psnr
                max_ssim = ssim
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))
        
        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t/prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--name", default = None)
    parser.add_argument("--tag", default = None)
    parser.add_argument("--gpu", default = 0)
    parser.add_argument("--bsize", default = None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
        if args.bsize is not None:
            config['train_bsize'] = int(args.bsize)
        print("config loaded")
    
    save_name = args.name
    if save_name is None:
        # WTF ?
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
    

    
    




