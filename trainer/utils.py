import glob
import os
import random
import warnings

import torch

import wandb
from dataset.cifar100 import CLIPDataset
from distributed import broadcast_object


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def logging(x_name, x_value, y_name, y_value, args):
    if args.wandb:
        wandb.define_metric(x_name)
        wandb.define_metric(y_name, step_metric=x_name)
        wandb.log({
            x_name: x_value,
            y_name: y_value
        })


def valid_split(Dataset, valid_rate):
    idx = Dataset.idx
    text = Dataset.text
    train_len = int(len(idx) * (1 - valid_rate))
    split_idx = random.sample(range(len(idx)), train_len)
    train_idx, val_idx, train_text, val_text = [], [], [], []
    for i, index in enumerate(idx):
        if i in split_idx:
            train_idx.append(index)
            train_text.append(text[i])
        else:
            val_idx.append(index)
            val_text.append(text[i])

    return CLIPDataset(Dataset.data, train_text, train_idx), CLIPDataset(Dataset.data, val_text, val_idx)


def is_master(args):
    if (not args.distributed) or args.rank == 0:
        return True
    else:
        return False


def get_ckpt_save_path(args, task, end='pt'):
    CHECKPOINT_NAME = f"task{task}.pt"
    if end == 'pth':
        CHECKPOINT_NAME += 'h'
    path = os.path.join(args.save_base_path, args.name,
                        "checkpoints_" + CHECKPOINT_NAME)
    return path


def resume(args, task, model, optimizer=None, scaler=None):
    start_epoch = 0
    if is_master(args):
        resume_from = get_ckpt_save_path(args, task)
        if not os.path.exists(resume_from):
            warnings.warn('Warning: No ckpt to resume')
            return model, optimizer, scaler, start_epoch

    if args.distributed:
        # sync found checkpoint path to all ranks
        resume_from = broadcast_object(args, resume_from)
    checkpoint = torch.load(resume_from, map_location='cpu')
    if 'epoch' in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)

    return model, optimizer, scaler, start_epoch
