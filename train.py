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
# WIP
from SSIM import SSIM
# from test import eval_psnr

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
                        shuffle = True) #, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=spec['batch_size'], 
                        shuffle = False) #, pin_memory=True)
    
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
    
def train(train_loader, model, optimizer):
    model.train()
    # find document reference
    # adjust others loss_fn to compared with same model
    L1_loss = nn.L1Loss()
    L2_loss = nn.MSELoss()
    train_loss = utils.Averager()

    """WIP"""
    for batch in tqdm(train_loader, desc = "train"):
        batch['lr_image'] = batch['lr_image'].cuda()
        batch['hr_coord'] = batch['hr_coord'].cuda()
        batch['cell'] = batch['cell'].cuda()
        
        
        # batch['lr_image']
        # batch['hr_image']
        # batch['hr_coord']
        # batch['focal_length']
        # batch['cell']

        ipdb.set_trace()
        pred_hr, pred_focal = model(batch['lr_image'], batch['hr_coord'], cell = batch['cell'])
    
    sr_loss = L1_loss(pred_hr, batch['hr_image'])
    focal_loss = L2_loss(pred_focal, batch['focal_length'])



    # loss = loss_fn
    return None



def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    train_loader, val_loader = make_data_loader(config.get('dataset'))

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    ipdb.set_trace()

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max+1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

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
            # func to calculate psnr (not yet implemented)
            val_res = eval_psnr(val_loader, model)

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                max_val_v = val_res
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
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
        print("config loaded")
    
    save_name = args.name
    if save_name is None:
        # WTF ?
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
    

    
    




