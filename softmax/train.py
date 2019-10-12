import os
import argparse
import numpy as np
import pandas as pd
import logging as log
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import settings
from loader import get_train_val_loaders, get_test_loader, get_frame_train_loader
#from mix_loader import get_train_val_loaders, get_test_loader

from models import create_model
from torch.nn import DataParallel
from tqdm import tqdm
from torch.nn import DataParallel
from utils import accuracy
#from apex import amp
from radam import RAdam

MODEL_DIR = settings.MODEL_DIR

c = nn.CrossEntropyLoss(reduction='none')

def _reduce_loss(loss):
    #print('loss shape:', loss.shape)
    return loss.sum() / loss.shape[0]

def criterion(output, target):
    return _reduce_loss(c(output, target))

def train(args):
    print('start training...')
    model, model_file = create_model(args)
    train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, val_batch_size=args.val_batch_size)
    #train_loader, val_loader = get_frame_train_loader(batch_size=args.batch_size, val_batch_size=args.val_batch_size)
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    elif args.optim == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name

    #model=model.train()

    best_f2 = 0.
    best_key = 'top1'

    print('epoch |    lr     |       %        |  loss  |  avg   |  loss  |  top1   |  top10  |  best  | time |  save |')

    if not args.no_first_val:
        val_metrics = validate(args, model, val_loader)
        print('val   |           |                |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} |       |        |'.format(
            val_metrics['valid_loss'], val_metrics['top1'], val_metrics['top10'], val_metrics[best_key] ))

        best_f2 = val_metrics[best_key]

    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_f2)
    else:
        lr_scheduler.step()

    train_iter = 0

    for epoch in range(args.start_epoch, args.num_epochs):
        #train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, val_batch_size=args.val_batch_size, val_num=args.val_num)

        train_loss = 0

        current_lr = get_lrs(optimizer)
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            #if train_loader.seg:
            rgb, audio, labels = [x.cuda() for x in data]
            #else:
            #    rgb, audio, labels = data[0].cuda(), data[2].cuda(), data[4].cuda()
            
            output = model(rgb, audio)
            
            loss = criterion(output, labels)
            batch_size = rgb.size(0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()

            train_loss += loss.item()
            print('\r {:4d} | {:.7f} | {:06d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                if isinstance(model, DataParallel):
                    torch.save(model.module.state_dict(), model_file+'_latest')
                else:
                    torch.save(model.state_dict(), model_file+'_latest')

                val_metrics = validate(args, model, val_loader)
                
                _save_ckp = ''
                if args.always_save or val_metrics[best_key] > best_f2:
                    best_f2 = val_metrics[best_key]
                    if isinstance(model, DataParallel):
                        torch.save(model.module.state_dict(), model_file)
                    else:
                        torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |  {:4s} |'.format(
                    val_metrics['valid_loss'], val_metrics['top1'], val_metrics['top10'], best_f2,
                    (time.time() - bg) / 60, _save_ckp))

                model.train()
                if args.lrs == 'plateau':
                    lr_scheduler.step(best_f2)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    #del model, optimizer, lr_scheduler

def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(args, model: nn.Module, valid_loader):
    model.eval()
    total_loss, top1_corrects, top10_corrects, total_num = 0., 0, 0, 0

    with torch.no_grad():
        for rgb, audio, labels in valid_loader:
            rgb, audio, labels = rgb.cuda(), audio.cuda(), labels.cuda()
            output = model(rgb, audio)
            #loss = criterion(output, labels)
            loss = c(output, labels).sum()

            top1, top10 = accuracy(F.softmax(output, 1), labels)
            top1_corrects += top1
            top10_corrects += top10
            total_num += len(rgb)

            total_loss += loss.item()

    metrics = {}
    metrics['valid_loss'] = total_loss / total_num
    metrics['top1'] = top1_corrects / total_num
    metrics['top10'] = top10_corrects / total_num

    #print(metrics)
    return metrics


def pred_model_output(model, loader):
    model.eval()
    outputs = []
    labels = []
    with torch.no_grad():
        for rgb, audio in tqdm(loader, total=loader.num // loader.batch_size):
            rgb, audio = rgb.cuda(), audio.cuda()
            output = model(rgb, audio)
            outputs.append(F.softmax(output,1).cpu())
    outputs = torch.cat(outputs).numpy()
    print(outputs.shape)
    return outputs

def predict(args):
    model, _ = create_model(args)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    test_loader = get_test_loader(batch_size=args.val_batch_size, dev_mode=args.dev_mode)
    probs = pred_model_output(model, test_loader)

    print(probs.shape)
    print(probs[:2])

    np.save(args.out, probs)

    #create_submission(args, scores)

def create_submission(args, scores):
    df = pd.read_csv(os.path.join(settings.DATA_DIR, 'test.csv'))
    df['prediction'] = scores

    df.to_csv(args.sub_file, header=True, index=False, columns=['id', 'prediction'])


def mean_df(args):
    df_files = args.mean_df.split(',')
    print(df_files)
    dfs = []
    for fn in df_files:
        dfs.append(pd.read_csv(fn))
    if args.weights is None:
        w = np.array([1] * len(dfs))
    else:
        w = np.array([int(x) for x in args.weights.split(',')])
    w = w / w.sum()
    print('w:', w)

    assert len(w) == len(dfs)

    df_sub = pd.read_csv(os.path.join(settings.DATA_DIR, 'test.csv'))

    preds = None
    for df, weight in zip(dfs, w):
        if preds is None:
            preds = df.prediction.values * weight
        else:
            preds += df.prediction.values * weight

    df_sub['prediction'] = preds
    df_sub.to_csv(args.sub_file, header=True, index=False, columns=['id', 'prediction'])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='learning rate')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=4096, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=4096, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=100, type=int, help='start epoch')
    parser.add_argument('--num_epochs', default=100, type=int, help='epoch')
    parser.add_argument('--optim', default='RAdam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=3, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.6, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=8, type=int, help='lr scheduler patience')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_model.pth',help='check point file name')
    parser.add_argument('--out', type=str, default='sub1.csv')
    parser.add_argument('--mean_df', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    parser.add_argument('--nlayers', default=3, type=int, help='layers')
    
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)

    if args.mean_df:
        mean_df(args)
    elif args.predict:
        predict(args)
    else:
        train(args)
