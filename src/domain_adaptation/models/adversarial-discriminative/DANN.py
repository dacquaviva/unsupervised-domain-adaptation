"""
Rev_Grad/DANN:
Unsupervised Domain Adaptation by Backpropagation, https://arxiv.org/pdf/1409.7495.pdf
Domain-adversarial training of neural networks, https://arxiv.org/pdf/1505.07818.pdf
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
from src.domain_adaptation.modules.classifier_network import ClassifierNetwork
from src.domain_adaptation.modules.gradient_reversal_layer import GradientReversalFn
from src.domain_adaptation.modules.adversarial_network import AdversarialNetwork
from utils.data.augmentation import DataAugmentation

class Dann(pl.LightningModule):

    def __init__(self,
            backbone : str,
            adversarial_network_width: int, 
            source_classifier_width: int,
            n_class : int,
            optim: torch.optim ,
            num_sample_test_embeddings_projector: int,
            augmentation,
            dropout_adversarial_network: bool,
            dropout_classifier_network: bool,
            **kwargs
            ):
        super().__init__()
        self.backbone = backbone
        self.n_class = n_class
        self.criterion = nn.CrossEntropyLoss()
        # self.source_train_accuracy = pl.metrics.Accuracy()
        # self.target_train_accuracy = pl.metrics.Accuracy()
        self.domain_source_train_accuracy = pl.metrics.Accuracy()
        self.domain_target_train_accuracy = pl.metrics.Accuracy()
        self.domain_source_val_accuracy = pl.metrics.Accuracy()
        self.domain_target_val_accuracy = pl.metrics.Accuracy()
        self.source_val_accuracy = pl.metrics.Accuracy()
        self.target_val_accuracy = pl.metrics.Accuracy()
        self.source_test_accuracy = pl.metrics.Accuracy()
        self.target_test_accuracy = pl.metrics.Accuracy()
        self.optim = optim
        self.p = 0
        self.num_sample_test_embeddings_projector = num_sample_test_embeddings_projector
        self.augmentation = DataAugmentation(**augmentation)
        if backbone == 'resnet50':
            self.backbone = Resnet50()
            out_features = self.backbone.out_dimension()
        elif backbone == 'vit':
            self.backbone = ViT()
            out_features = self.backbone.out_dimension()
        
        self.classifier_layer = ClassifierNetwork(out_features, source_classifier_width, n_class, dropout=dropout_classifier_network)    
            
        self.domain_classifier = AdversarialNetwork(out_features, adversarial_network_width, dropout=dropout_adversarial_network)
   
    def forward(self, source, target, grl_lambda=1.0):
        source_embeddings = self.backbone(source)
        target_embeddings = self.backbone(target)
        source_clf = self.classifier_layer(source_embeddings)
        source_reverse_embeddings = GradientReversalFn.apply(source_embeddings, grl_lambda)
        domain_source_clf = self.domain_classifier(source_reverse_embeddings)
        target_reverse_embeddings = GradientReversalFn.apply(target_embeddings, grl_lambda)
        domain_target_clf = self.domain_classifier(target_reverse_embeddings)
        return source_clf, domain_source_clf, domain_target_clf
    
    def predict(self, x):
        embeddings = self.backbone(x)
        clf = self.classifier_layer(embeddings)
        return clf, embeddings
    
    def compute_grl_lambda(self, batch_idx):
        #In the beginning of training, the domain loss is extremely noisy since the CNN features are not good yet. Therefore, lambda is gradulaly changed from 0 to 1 in the course of training.
        #p refers to the tranining progress changing linearly from 0 to 1.
        max_batches = self.trainer.num_training_batches
        self.p = float(batch_idx + self.current_epoch * max_batches) / self.trainer.max_epochs / max_batches #Set p as instance variable to use it in lr scheduler
        grl_lambda = 2. / (1. + np.exp(-10 * self.p)) - 1
        return grl_lambda
            
    def training_step(self, batch, batch_idx):
        source_batch, source_labels = batch[0]
        source_batch = self.augmentation(source_batch) #Kornia augmentation
        target_batch, target_labels = batch[1]
        target_batch = self.augmentation(target_batch) #Kornia augmentation
        grl_lambda = self.compute_grl_lambda(batch_idx)
        
        #Forward
        source_clf, domain_source_clf, domain_target_clf = self(source_batch, target_batch, grl_lambda)
        #Loss labels source classification
        source_loss_cls = self.criterion(source_clf, source_labels.squeeze().long())
        #Loss source domain classification
        domain_source_labels = torch.zeros(len(source_batch)).type_as(source_batch) # generate source domain labels
        domain_source_clf_loss = self.criterion(domain_source_clf, domain_source_labels.long())
        self.domain_source_train_accuracy(domain_source_clf.softmax(dim=-1), domain_source_labels.int())
        #Loss target domain classification
        domain_target_labels = torch.ones(len(target_batch)).type_as(target_batch) # generate target domain labels
        domain_target_clf_loss = self.criterion(domain_target_clf, domain_target_labels.long())
        self.domain_target_train_accuracy(domain_target_clf.softmax(dim=-1), domain_target_labels.int())
        #Total loss
        loss = source_loss_cls + domain_source_clf_loss + domain_target_clf_loss
        #Log steps Progress bar only
        self.log('Source_Loss', source_loss_cls, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('Domain_Source_Loss', domain_source_clf_loss,  on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('Domain_Target_Loss', domain_target_clf_loss,  on_step=True, on_epoch=False, prog_bar=True, logger=False)
        #Log epochs Logger only
        self.log('Training/Loss/Source', source_loss_cls, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Training/Loss/Domain_Source', domain_source_clf_loss,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Training/Loss/Domain_Target', domain_target_clf_loss,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Training/Loss/Total', loss,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Training/Accuracy/Domain_Source', self.domain_source_train_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Training/Accuracy/Domain_Target', self.domain_target_train_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # #Source Train Accuracy
        # source_clf, _ = self.predict(source_batch)   
        # self.source_train_accuracy(source_clf.softmax(dim=-1) , source_labels)
        # self.log('Training/Accuracy/Source', self.source_train_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True) # It automatically compute avg accuracy end of epoch
        
        # target_clf, _ = self.predict(target_batch)
        # self.target_train_accuracy(target_clf.softmax(dim=-1), target_labels)
        # self.log('Training/Accuracy/Target', self.target_train_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)  # It automatically compute avg accuracy end of epoch
        
        #Every epoch Log first batch Embeddings Distribution to TensorBoard
        # if batch_idx == 0:
        #     histogram.add_histogram_embeddings(self, source_batch, target_batch)
            
        return loss
    
    # #Log parameters distribution at the end of every epoch to TensorBoard
    # def training_epoch_end(self, outputs):
    #     histogram.add_histogram_parameters(self)#Debug
        
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
        
        #Compute Losses
        
        #Forward
        source_clf, domain_source_clf, domain_target_clf = self(source_batch, target_batch)
        #Loss labels source classification
        source_loss_cls = self.criterion(source_clf, source_labels.squeeze().long())
        #Loss source domain classification
        domain_source_labels = torch.zeros(len(source_batch)).type_as(source_batch) # generate source domain labels
        domain_source_clf_loss = self.criterion(domain_source_clf, domain_source_labels.long())
        self.domain_source_val_accuracy(domain_source_clf.softmax(dim=-1), domain_source_labels.int())
        #Loss target domain classification
        domain_target_labels = torch.ones(len(target_batch)).type_as(target_batch) # generate target domain labels
        domain_target_clf_loss = self.criterion(domain_target_clf, domain_target_labels.long())
        self.domain_target_val_accuracy(domain_target_clf.softmax(dim=-1), domain_target_labels.int())
        #Total loss
        loss = source_loss_cls + domain_source_clf_loss + domain_target_clf_loss
        #Log epochs Logger only
        self.log('Validation/Loss/Source', source_loss_cls, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Validation/Loss/Domain_Source', domain_source_clf_loss,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Validation/Loss/Domain_Target', domain_target_clf_loss,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Validation/Loss/Total', loss,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Validation/Accuracy/Domain_Source', self.domain_source_val_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Validation/Accuracy/Domain_Target', self.domain_target_val_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
            
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
        
        #Add graph to tensorboard
        if batch_idx==0 and  isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_graph(self, (source_batch, target_batch))
                    
    def on_test_end(self):
        labels = self.trainer.test_dataloaders[0].dataset.datasets[0].classes
        #Log Confusion Matrix to TensorBoard
        self.source_confusion_matrix_tensorboard.log_confusion_matrix(labels, "Source")
        self.target_confusion_matrix_tensorboard.log_confusion_matrix(labels, "Target")
        if self.num_sample_test_embeddings_projector > 0. :
            # #Log embeddings to TensorBoard Projector
            self.projector_tensorboard.log_embeddings(labels, num_samples=self.num_sample_test_embeddings_projector)#percentage indicates what percentage of data is loaded to tensorboard
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optim.opt, params=self.parameters())
        lambda1 = lambda epoch: 1/((1 + self.optim.scheduler.gamma* self.p)**self.optim.scheduler.beta)
        scheduler = {"scheduler": lr_scheduler.LambdaLR(optimizer, lambda1), 'name': 'optimizer_scheduler'}
        return [optimizer], [scheduler]
        return optimizer
        
        
    
