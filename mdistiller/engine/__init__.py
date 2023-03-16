from .trainer_IA import IATrainer
from .trainer_an import ANTrainer
from .trainer import BaseTrainer,CRDTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "an": ANTrainer,
    "IAKD": IATrainer
}



