# Finetune CLIP for transfer learning and continual learning
Finetune CLIP for transfer learning, continual learning, zero-shot learning and continual zero-shot learning. 

We finetune CLIP with constrastive loss. We use AdamW optimizer and CosineAnnealingWarmup scheduler. Learning rate is set to 7.5e-6 and we finetune 10 epochs. 

### Installation
```angular2html
conda env create -f environment.yaml
```
### Dataset list
- aircraft: Supported by torchvision
- birdsnap: Downloaded from the official website
- cars: Supported by HuggingFace
- cifar: Supported by torchvision
- cub: Supported by HuggingFace
- gtsrb: Supported by torchvision

### Metrics
For Transfer Learning
- Whole task accuracy

For continual learning 
- Class-incremental Accuracy
- Task/Domain incremental Accuracy

For zero-shot leanring
- ImageNet zero-shot accuracy

For continual zero-shot learning
- Whole set accuracy
- Unseen accuracy

### Method 
- Frozen: zero-shot evaluation of CLIP
- Finetune: finetune visual and text tower of CLIP
- Finetunevisual: only finetune the visual tower of CLIP
- FinetuneFFN: only finetune the FFN layers of the visual tower of CLIP
- FinetuneCproj: only finetune the second FFN layers of the visual tower of CLIP
- FinetuneTextCproj: only finetune the second FFN layers of the text tower of CLIP
- FinetuneCprojboth: only finetune the second FFN layers of both towers of CLIP

###  Usage
Transfer Learning
```angular2html
python main.py dataset=${DATASET} method=${METHOD} joint=True
```

Continual learning, zero-shot learning and continual zero-shot learning
```angular2html
python main.py dataset=${DATASET} method=${METHOD}
```

### Customize Finetuning
- Change the unfrozen part: edit the function `unfreeze_model` in `finetune.py`
- Change the loss function: edit the function `compute_loss` in `finetune.py`


### Citation
```
@misc{finetuneclip,
  author = {Wenxuan Zhang},
  title = {Finetune CLIP for transfer learning and continual learning},
  year = {2013},
  publisher = {GitHub},
  url = {https://github.com/wx-zhang/FinetuneCLIP},
}
```
