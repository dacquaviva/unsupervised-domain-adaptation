import hydra
import os
import torch
from pytorch_lightning import LightningModule

import yaml
import glob
from src.domain_adaptation.models.normal.SOURCE_ONLY import Source_only as Model


def init_model(model_path):
    with open(os.path.join(model_path, ".hydra/config.yaml"), 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    model_checkpoint_path = os.path.join(model_path, "checkpoints", "last.ckpt")
    model = Model.load_from_checkpoint(model_checkpoint_path,  **conf["model"], n_class=conf["datamodule"]["n_class"], optim=conf["optim"], augmentation=conf["datamodule"]["data_augmentation"], num_sample_test_embeddings_projector=0)

    return model
