
import os
import random
import time
from argparse import Namespace


import hydra
import numpy as np
import omegaconf
import torch

import wandb
from clip import clip
from dataset.aircraft import SplitAircraft
from dataset.birdsnap import SplitBirdsnap
from dataset.cars import SplitCars
from dataset.cifar100 import SplitCifar100
from dataset.cub import CUB
from dataset.gtsrb import SplitGTSRB
from trainer import METHOD


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(args):
    args = omegaconf.OmegaConf.to_container(args)
    args = Namespace(**args)

    start = time.time()

    random_seed(args.seed)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # get the name of the experiments and setup logging
    if args.name is None:
        args.name = '-'.join([
            args.method,
            args.dataset,
            os.environ.get("SLURM_JOB_ID", ""),
        ])
    log_base_path = os.path.join(args.logs, args.name)

    os.makedirs(log_base_path, exist_ok=True)
    args.log_path = log_base_path
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    if args.wandb:

        wandb.init(
            # put your wandb initiation

        )

    print("{}".format(args).replace(', ', ',\n'))

    # set up model
    model, transform = clip.load(
        args.model, download_root='./clip_models/', args=args)

    args.hidden_size = model.visual.proj.shape[0]
    args.visual_layers = len(model.visual.transformer.resblocks)

    if args.dataset == 'cifar100':
        dataset = SplitCifar100(args, args.data, transform)
    elif args.dataset == 'cars':
        dataset = SplitCars(args, transform=transform)
    elif args.dataset == 'cub':
        dataset = CUB(args, transform=transform)
    elif args.dataset == 'aircraft':
        dataset = SplitAircraft(args, transform=transform)
    elif args.dataset == 'birdsnap':
        dataset = SplitBirdsnap(args, transform=transform)
    elif args.dataset == 'gtsrb':
        dataset = SplitGTSRB(args, transform=transform)
    else:
        raise ValueError

    args.num_classes = dataset.num_classes
    args.num_tasks = dataset.num_tasks
    args.scenario = dataset.scenario
    Trainer = METHOD[args.method](args)

    for task in range(dataset.num_tasks):
        if args.sweep and task == 3:
            break
        print(f'Train task {task}')
        if args.evaluation:
            Trainer.only_evaluation(model, dataset, task)
            break
        Trainer.train(model, dataset, task)
        Trainer.evaluation(model, dataset, task)
        Trainer.save_checkpoint(model, task, args)

    print(f'Total training time in hours: {(time.time() - start) / 3600: .3f}')

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
