"""
Normal model (Source only)
Model only trained on source domain. Then make predictions on target domain.
"""
import torch
import torch.nn as nn
import torchvision

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pytorch_lightning as pl
import hydra

from utils.tensorboard.confusion_matrix_tensorboard import ConfusionMatrixTensorBoard
from utils.tensorboard.projector_tensorboard import ProjectorTensorBoard
import utils.tensorboard.histogram_tensorboard as histograms
from src.domain_adaptation.backbones.resnet50 import Resnet50
from src.domain_adaptation.backbones.vit import ViT
from src.domain_adaptation.modules.classifier_network import ClassifierNetwork
from utils.data.augmentation import DataAugmentation

class Source_only(pl.LightningModule):

    def __init__(self,
            backbone : str,
            source_classifier_width: int,
            n_class : int,
            optim: torch.optim,
            num_sample_test_embeddings_projector: int,
            augmentation,
            dropout_classifier_network=True,
            **kwargs
            ):
        super().__init__()
        self.backbone = backbone
        self.n_class = n_class
        self.criterion = nn.CrossEntropyLoss()
        self.source_val_accuracy = pl.metrics.Accuracy()
        self.target_val_accuracy = pl.metrics.Accuracy()
        self.source_test_accuracy = pl.metrics.Accuracy()
        self.target_test_accuracy = pl.metrics.Accuracy()
        self.optim = optim
        self.num_sample_test_embeddings_projector = num_sample_test_embeddings_projector
        self.augmentation = DataAugmentation(**augmentation)
        if backbone == 'resnet50':
            self.backbone = Resnet50()
            out_features = self.backbone.out_dimension()
        elif backbone == 'vit':
            self.backbone = ViT()
            out_features = self.backbone.out_dimension()
        
        self.classifier_layer = ClassifierNetwork(out_features, source_classifier_width, n_class, dropout=dropout_classifier_network)      
        
    def forward(self, source):
        embeddings = self.backbone(source)
        source_clf = self.classifier_layer(embeddings)
        return source_clf, embeddings
    
    def predict(self, source):
        embeddings = self.backbone(source)
        source_clf = self.classifier_layer(embeddings)
        return source_clf, embeddings
    
    def training_step(self, batch, batch_idx):
        source_batch, source_labels = batch[0]
        source_batch = self.augmentation(source_batch) #Kornia augmentation
        source_clf, _  = self(source_batch)
        loss = self.criterion(source_clf, source_labels.squeeze().long())
        #Log epochs Logger only
        self.log('Training/Loss/Total Loss', loss,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):
        #Source Val Accuracy
        source_batch, source_labels = batch[0]
        source_clf, _ = self.predict(source_batch)   
        self.source_val_accuracy(source_clf.softmax(dim=-1) , source_labels)
        self.log('Validation/Accuracy/Source', self.source_val_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True) # It automatically compute avg accuracy end of epoch
        
        #Target Val Accuracy
        target_batch, target_labels = batch[1]
        target_clf, _ = self.predict(target_batch)
        self.target_val_accuracy(target_clf.softmax(dim=-1), target_labels)
        self.log('Validation/Accuracy/Target', self.target_val_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)  # It automatically compute avg accuracy end of epoch
        
        #Compute Losses
        clf_loss = self.criterion(source_clf, source_labels.squeeze().long())
        self.log('Validation/Loss/Source', clf_loss,  on_step=False, on_epoch=True, prog_bar=False, logger=True)

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
        self.log('Test/Accuracy/Source', self.source_test_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True) # It automatically compute avg accuracy end of epoch
        
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
        self.log('Test/Accuracy/Target', self.target_test_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)  # It automatically compute avg accuracy end of epoch
        
        #Log classification Target coufusion matrix combining all batches first
        self.target_confusion_matrix_tensorboard.add_y_true(target_labels.tolist())
        _, target_preds = torch.max(target_clf.softmax(dim=-1), -1)
        self.target_confusion_matrix_tensorboard.add_y_pred(target_preds.tolist())
        if self.num_sample_test_embeddings_projector > 0. :
            #Log Target embeddings combining all batches first
            self.projector_tensorboard.add_target_embeddings(target_embeddings, target_labels, target_batch)
            
        #Add graph to tensorboard
        if batch_idx==0 and  isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_graph(self, (source_batch))
            
    def on_test_end(self):
        labels = self.trainer.test_dataloaders[0].dataset.datasets[0].classes
        #Log Confusion Matrix to TensorBoard
        self.source_confusion_matrix_tensorboard.log_confusion_matrix(labels, "Source")
        self.target_confusion_matrix_tensorboard.log_confusion_matrix(labels, "Target")
        if self.num_sample_test_embeddings_projector > 0. :
            # #Log embeddings to TensorBoard Projector
            self.projector_tensorboard.log_embeddings(labels, num_samples=self.num_sample_test_embeddings_projector)#percentage indicates what percentage of data is loaded to tensorboard
        
    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optim.opt, params=self.parameters())
    
    