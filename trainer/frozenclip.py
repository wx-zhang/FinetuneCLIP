from dataset.imagenet import zeroshot_classifier
from clip.clip import tokenize
from dataset.imagenet import ImageNet
from metric import AverageMeter

import torch
import wandb
import statistics
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import numpy as np



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


class FrozenCLIP(object):
    def __init__(self, args):
        self.args = args
        self.num_classes = args.num_classes
        self.metric = [0] * args.num_tasks
        self.held_out = 0

    def only_evaluation(self, model, dataset, task):
        pass

    def train(self, model, dataset, task):
        pass

    def zero_shot_evaluation(self, model, transform):
        testset = ImageNet(transform)
        metric = AverageMeter()
        zeroshot_weights = zeroshot_classifier(testset.classes, model)
        test_dataloader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        for image, label in tqdm(test_dataloader, desc=f"Evaluation for ImageNet Validation Set",
                                 total=len(test_dataloader)):
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights
            acc = accuracy(logits, label)[0]
            metric.update(acc, image.size(0))
        return metric.avg.item()

    def evaluation(self, model, dataset, task):

        metric = AverageMeter()

        testset = dataset.get_dataset(task, is_train=False)

        if self.args.scenario == 'class_incremental':

            if hasattr(dataset, 'classifier'):
                text_features_full = dataset.classifier(
                    dataset.class_name_full, model)

            else:

                text_inputs_full = torch.cat(
                    [tokenize(f"a photo of a {c}") for c in dataset.class_name_full]).cuda()
                with torch.no_grad():
                    text_features_full = model.encode_text(text_inputs_full)
                    text_features_full /= text_features_full.norm(
                        dim=1, keepdim=True)

            if task < dataset.num_tasks - 1:
                unseen_class_idx = torch.Tensor(np.concatenate(
                    dataset.task_classes[task + 1:], axis=None)).to(torch.long)

                text_features = text_features_full.clone().detach()
                text_features[unseen_class_idx] = 0
            else:
                text_features = text_features_full.clone().detach()

        test_dataloader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        for (image, label, _) in tqdm(test_dataloader, desc=f"Evaluation for {task}",
                                      total=len(test_dataloader)):
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T

            acc = accuracy(logits, label, topk=(1,))[0]
            metric.update(acc.item(), image.size(0))

        if task == 0:
            self.held_out = self.held_out_evaluation(model, dataset.transform) if not (
                self.args.debug or self.args.sweep) else 0

        print(f'Accuracy for Task {task}: {metric.avg}')
        print(f'Held out accuracy {self.held_out: .2f}')
        self.metric[task] = metric.avg
        if task == self.args.num_tasks - 1:
            print(f'Average Accuracy: {statistics.mean(self.metric)}')

        if self.args.report_to:
            logging('task', task, 'accuracy', metric.avg, self.args)
            logging('task', task, 'average accuracy',
                    statistics.mean(self.metric[:task+1]), self.args)
            logging('task', task, 'held out accuracy',
                    self.held_out, self.args)

    def save_checkpoint(self, model, task, args):
        pass
