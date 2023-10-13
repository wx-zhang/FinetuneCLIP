
from trainer.finetune import FinetuneCLIP, FinetuneFFN, FinetuenProj, FinetuneTextProj, FinetuenProjTV
from trainer.frozenclip import FrozenCLIP
METHOD = {'FrozenCLIP': FrozenCLIP,
          'Finetune': FinetuneCLIP,
          'finetunevisual': FinetuneCLIP,
          'FinetuneFFN': FinetuneFFN,
          'FinetuneCproj': FinetuenProj,
          'FinetuneCprojboth': FinetuenProjTV,
          'FinetuneTextCproj': FinetuneTextProj,
          }
