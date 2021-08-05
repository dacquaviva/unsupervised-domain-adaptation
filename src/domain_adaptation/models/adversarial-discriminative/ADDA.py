"""
ADDA:
Adversarial Discriminative Domain Adaptation, https://arxiv.org/pdf/1702.05464.pdf
"""
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Function
import torch.optim.lr_scheduler as lr_scheduler


from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pytorch_lightning as pl

import hydra
import numpy as np

from utils.tensorboard.confusion_matrix_tensorboard import ConfusionMatrixTensorBoard
from utils.tensorboard.projector_tensorboard import ProjectorTensorBoard
import utils.tensorboard.histogram_tensorboard as histogram
from src.domain_adaptation.backbones.resnet50 import Resnet50
from src.domain_adaptation.backbones.vit import ViT
from utils.model.init_model import init_model
from pytorch_lightning import Trainer
from utils.data.augmentation import DataAugmentation
from src.domain_adaptation.modules.adversarial_network import AdversarialNetwork

class Adda(pl.LightningModule):

    def __init__(self, source_model_restore_path: str, backbone: str,  optim: torch.optim, n_class : int, adversarial_network_width: int, augmentation, num_sample_test_embeddings_projector, dropout_adversarial_network: bool, **kwargs):
        super().__init__()
        
        self.source_model = init_model(source_model_restore_path)
        self.source_encoder, self.source_classifier = self.source_model.backbone, self.source_model.classifier_layer
        self.target_encoder = init_model(source_model_restore_path).backbone
        out_features = self.source_model.backbone.out_dimension()
        self.discriminator = AdversarialNetwork(out_features, adversarial_network_width, dropout=dropout_adversarial_network) 
        self.optim = optim
        self.criterion = nn.CrossEntropyLoss()
        self.source_val_accuracy = pl.metrics.Accuracy()
        self.target_val_accuracy = pl.metrics.Accuracy()
        self.source_test_accuracy = pl.metrics.Accuracy()
        self.target_test_accuracy = pl.metrics.Accuracy()
        self.num_sample_test_embeddings_projector = num_sample_test_embeddings_projector
        self.augmentation = DataAugmentation(**augmentation)

    def forward(self, x):
        embeddings =  self.target_encoder(x)
        clf = self.source_classifier(embeddings)
        return clf, embeddings
    
    def predict(self, x):
        embeddings =  self.target_encoder(x)
        clf = self.source_classifier(embeddings)
        return clf, embeddings
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        source_batch, _ = batch[0]
        source_batch = self.augmentation(source_batch) #Kornia augmentation
        target_batch, _ = batch[1]
        target_batch = self.augmentation(target_batch) #Kornia augmentation
        
        ###########################
        # train discriminator     #
        ###########################
        if optimizer_idx == 0:
            # extract and concat features
            source_embeddings = self.source_encoder(source_batch)
            target_embeddings = self.target_encoder(target_batch)
            concat_embeddings = torch.cat((source_embeddings, target_embeddings), 0)
            
            # predict on discriminator
            concat_pred = self.discriminator(concat_embeddings.detach())
            
            # prepare real and fake label
            source_label = torch.ones(source_embeddings.size(0)).type_as(source_embeddings)
            target_label = torch.zeros(target_embeddings.size(0)).type_as(target_embeddings)
            concat_label = torch.cat((source_label, target_label), 0).long()
            
            # compute loss for critic
            loss_discriminator = self.criterion(concat_pred, concat_label)
            
            #Log steps Progress bar only
            # self.log('Discriminator_Loss', loss_discriminator, on_step=True, on_epoch=False, prog_bar=True, logger=False)
            
            #Log epochs Logger only
            self.log('Training/Loss/Discriminator', loss_discriminator, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            
            return loss_discriminator
        
        ############################
        # train target encoder     #
        ############################
        if optimizer_idx == 1:
            # extract and target features
            target_embeddings = self.target_encoder(target_batch)
            
            # predict on discriminator
            target_pred = self.discriminator(target_embeddings)
            
            # prepare fake labels
            target_label_fake = torch.ones(target_pred.size(0)).type_as(target_pred).long()
            
            # compute loss for target encoder
            loss_target = self.criterion(target_pred, target_label_fake)
            
            #Log steps Progress bar only
            # self.log('Target_Loss', loss_target, on_step=True, on_epoch=False, prog_bar=True, logger=False)
            
            #Log epochs Logger only
            self.log('Training/Loss/Target', loss_target, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            
            return loss_target
                
    def validation_step(self, batch, batch_idx):
        #Compute Accuracy
        
        #Source Val Accuracy
        source_batch, source_labels = batch[0]
        source_clf, _ = self.predict(source_batch)   
        self.source_val_accuracy(source_clf.softmax(dim=-1) , source_labels)
       
        #Target Val Accuracy
        target_batch, target_labels = batch[1]
        target_clf, _ = self.predict(target_batch)
        self.target_val_accuracy(target_clf.softmax(dim=-1), target_labels)
        
        self.log('Validation/Accuracy/Source', self.source_val_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True) # It automatically compute avg accuracy end of epoch
        self.log('Validation/Accuracy/Target', self.target_val_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)  # It automatically compute avg accuracy end of epoch
        
    def on_test_start(self):
        #Initialize Confusion Matrix
        self.source_confusion_matrix_tensorboard = ConfusionMatrixTensorBoard(self.logger)
        self.target_confusion_matrix_tensorboard = ConfusionMatrixTensorBoard(self.logger)
        if self.num_sample_test_embeddings_projector > 0. :
            #Initialize Projector TensorBoard
            self.projector_tensorboard = ProjectorTensorBoard(self.logger)
    
    def test_step(self, batch, batch_idx):
        source_batch, source_labels = batch[0]
        source_clf, source_embeddings = self.predict(source_batch)
        self.source_test_accuracy(source_clf.softmax(dim=-1), source_labels)
        self.log('Test/Accuracy/Source', self.source_test_accuracy) # It automatically compute avg accuracy end of epoch
        
        #Log classification Source coufusion matrix combining all batches first
        self.source_confusion_matrix_tensorboard.add_y_true(source_labels.tolist())
        _, source_preds = torch.max(source_clf.softmax(dim=-1), -1)
        self.source_confusion_matrix_tensorboard.add_y_pred(source_preds.tolist())
        if self.num_sample_test_embeddings_projector > 0. :
            #Log Source embeddings combining all batches first
            self.projector_tensorboard.add_source_embeddings(source_embeddings, source_labels, source_batch)
    
        target_batch, target_labels = batch[1]
        target_clf, target_embeddings = self.predict(target_batch)
        self.target_test_accuracy(target_clf.softmax(dim=-1), target_labels)
        self.log('Test/Accuracy/Target', self.target_test_accuracy)  # It automatically compute avg accuracy end of epoch
        
        #Log classification Target coufusion matrix combining all batches first
        self.target_confusion_matrix_tensorboard.add_y_true(target_labels.tolist())
        _, target_preds = torch.max(target_clf.softmax(dim=-1), -1)
        self.target_confusion_matrix_tensorboard.add_y_pred(target_preds.tolist())
        if self.num_sample_test_embeddings_projector > 0. :
            #Log Target embeddings combining all batches first
            self.projector_tensorboard.add_target_embeddings(target_embeddings, target_labels, target_batch)
        

            
    def on_test_end(self):
        labels = self.trainer.test_dataloaders[0].dataset.datasets[0].classes
        #Log Confusion Matrix to TensorBoard
        self.source_confusion_matrix_tensorboard.log_confusion_matrix(labels, "Source")
        self.target_confusion_matrix_tensorboard.log_confusion_matrix(labels, "Target")
        if self.num_sample_test_embeddings_projector > 0. :
            # #Log embeddings to TensorBoard Projector
            self.projector_tensorboard.log_embeddings(labels, num_samples=self.num_sample_test_embeddings_projector)#percentage indicates what percentage of data is loaded to tensorboard
        
    def configure_optimizers(self):
        discriminator_opt = hydra.utils.instantiate(self.optim.opt, params=self.discriminator.parameters())
        target_optimizer = hydra.utils.instantiate(self.optim.opt, params=self.target_encoder.parameters())
        return discriminator_opt, target_optimizer