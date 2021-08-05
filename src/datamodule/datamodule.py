import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
import os
from pytorch_lightning.trainer.supporters import CombinedLoader
from utils.data.preprocess import PreProcess
from src.datamodule.dataset import ConcatDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, source_domain_path: str , target_domain_path: str , batch_size: int, num_workers: int, validation_split: int, data_processing, **kwargs):
        super().__init__()
        self.source_domain_path = source_domain_path
        self.target_domain_path = target_domain_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.preprocess = PreProcess(**data_processing)

    def setup(self, stage= None):
        if "complete" not in self.source_domain_path and self.target_domain_path:            
            source_root_train = os.path.join(self.source_domain_path, "train")
            target_root_train = os.path.join(self.target_domain_path, "train")
            source_root_test = os.path.join(self.source_domain_path, "test")
            target_root_test = os.path.join(self.target_domain_path, "test")
        else:
            source_root_train = os.path.join(self.source_domain_path)
            target_root_train = os.path.join(self.target_domain_path)
            source_root_test = os.path.join(self.source_domain_path)
            target_root_test = os.path.join(self.target_domain_path)
                
        source_training = ImageFolder(root=source_root_train, transform=self.preprocess)
        target_training = ImageFolder(root=target_root_train, transform=self.preprocess)

        source_test = ImageFolder(root=source_root_test, transform=self.preprocess)
        target_test = ImageFolder(root=target_root_test, transform=self.preprocess)
        
        if  self.validation_split != 0.:
            source_size = len(source_training)
            source_train_size = int(source_size * (1 - self.validation_split))
            source_val_size = source_size - source_train_size
            source_training, source_val = random_split(source_training, [source_train_size, source_val_size])
            
            target_size = len(target_training)
            target_train_size = int(target_size * (1 - self.validation_split))
            target_val_size = target_size - target_train_size
            target_training, target_val = random_split(target_training, [target_train_size, target_val_size])
        else:
            source_val = source_test
            target_val = target_test
            
        self.concat_dataset_training = ConcatDataset(source_training, target_training)
        self.concat_dataset_val = ConcatDataset(source_val, target_val)
        self.concat_dataset_test = ConcatDataset(source_test, target_test)
    
    def train_dataloader(self):
        return DataLoader(self.concat_dataset_training, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.concat_dataset_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.concat_dataset_test, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True)# shuffle=True to log random embeddings to tensorboard
