"""
CDAN:
Conditional Adversarial Domain Adaptation, https://arxiv.org/pdf/1705.10667.pdf
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.domain_adaptation.backbones.resnet50 import Resnet50
from src.domain_adaptation.backbones.vit import ViT
import numpy as np
import hydra
import torch.optim.lr_scheduler as lr_scheduler
from utils.tensorboard.confusion_matrix_tensorboard import ConfusionMatrixTensorBoard
from utils.tensorboard.projector_tensorboard import ProjectorTensorBoard
import utils.tensorboard.histogram_tensorboard as histogram
from src.domain_adaptation.modules.gradient_reversal_layer import GradientReversalFn
from src.domain_adaptation.modules.adversarial_network import AdversarialNetwork
from src.domain_adaptation.modules.random_layer import RandomLayer
from src.domain_adaptation.modules.classifier_network import ClassifierNetwork
from utils.data.augmentation import DataAugmentation

class Cdan(pl.LightningModule):
    def __init__(self,
            backbone : str,
            n_class : int,
            random: bool,
            optim: torch.optim,
            num_sample_test_embeddings_projector: int,
            augmentation,
            adversarial_network_width: int,
            random_dim: int,
            source_classifier_width: int,
            method: str,
            trade_off: int,
            dropout_adversarial_network: bool,
            dropout_classifier_network: bool,
            **kwargs):
        
        super().__init__()
        self.backbone = backbone
        self.optim = optim
        self.num_sample_test_embeddings_projector = num_sample_test_embeddings_projector
        self.augmentation = DataAugmentation(**augmentation)
        self.random = random
        self.criterion = nn.CrossEntropyLoss()
        self.method = method
        self.source_val_accuracy = pl.metrics.Accuracy()
        self.target_val_accuracy = pl.metrics.Accuracy()
        self.source_test_accuracy = pl.metrics.Accuracy()
        self.target_test_accuracy = pl.metrics.Accuracy()
        self.p = 0
        self.trade_off = trade_off
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        
        if backbone == 'resnet50':
            self.backbone = Resnet50()
            out_features = self.backbone.out_dimension()
        elif backbone == 'vit':
            self.backbone = ViT()
            out_features = self.backbone.out_dimension()
            
        self.classifier_layer = ClassifierNetwork(out_features, source_classifier_width, n_class, dropout=dropout_classifier_network)        

        if self.random:
            self.random_layer = RandomLayer([out_features, n_class], random_dim)
            self.ad_net = AdversarialNetwork(random_dim, adversarial_width, dropout=dropout_adversarial_network)
        else:
            self.random_layer = None
            self.ad_net = AdversarialNetwork(out_features * n_class, adversarial_network_width, dropout=dropout_adversarial_network)
            
    def forward(self, x):
        embeddings = self.backbone(x)
        clf = self.classifier_layer(embeddings)
        return clf, embeddings
    
    def predict(self, x):
        embeddings = self.backbone(x)
        clf = self.classifier_layer(embeddings)
        return clf, embeddings

    def training_step(self, batch, batch_idx):
        max_batches = self.trainer.num_training_batches
        self.p = float(batch_idx + self.current_epoch * max_batches) / self.trainer.max_epochs / max_batches #Set p as instance variable to use it in lr scheduler
        coeff = calc_coeff(self.trainer.global_step, self.high, self.low, self.alpha, self.max_iter)
        source_batch, source_labels = batch[0]
        target_batch, _ = batch[1]
        batch = torch.cat((source_batch, target_batch), 0)
        batch = self.augmentation(batch) #Kornia augmentation
        outputs, feature  = self(batch)
        classifier_loss = self.criterion(outputs.narrow(0, 0, source_batch.size(0)), source_labels)
        softmax_out = nn.Softmax(dim=1)(outputs)
        
        if self.method == 'CDAN-E':
            entropy = Entropy(softmax_out)
            transfer_loss = CDAN([feature, softmax_out], self.ad_net, entropy, coeff, self.random_layer)
        elif self.method == 'CDAN':
            transfer_loss = CDAN([feature, softmax_out], self.ad_net, None, coeff,  self.random_layer)
        
        total_loss = self.trade_off * transfer_loss + classifier_loss
                
        self.log('Training/Loss/Total', total_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return total_loss
    
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
        target_batch, target_labels = batch[1]

            
        source_clf, source_embeddings = self.predict(source_batch)
        target_clf , target_embeddings = self.predict(target_batch)
        
        self.source_test_accuracy(source_clf.softmax(dim=-1), source_labels)
        self.target_test_accuracy(target_clf.softmax(dim=-1), target_labels)
        self.log('Test/Accuracy/Source', self.source_test_accuracy) # It automatically compute avg accuracy end of epoch
        self.log('Test/Accuracy/Target', self.target_test_accuracy)  # It automatically compute avg accuracy end of epoch
        
        #Log classification Source coufusion matrix combining all batches first
        self.source_confusion_matrix_tensorboard.add_y_true(source_labels.tolist())
        _, source_preds = torch.max(source_clf.softmax(dim=-1), -1)
        self.source_confusion_matrix_tensorboard.add_y_pred(source_preds.tolist())
        if self.num_sample_test_embeddings_projector > 0. :
            #Log Source embeddings combining all batches first
            self.projector_tensorboard.add_source_embeddings(source_embeddings, source_labels, source_batch)
        
        #Log classification Target coufusion matrix combining all batches first
        self.target_confusion_matrix_tensorboard.add_y_true(target_labels.tolist())
        _, target_preds = torch.max(target_clf.softmax(dim=-1), -1)
        self.target_confusion_matrix_tensorboard.add_y_pred(target_preds.tolist())
        if self.num_sample_test_embeddings_projector > 0. :
            #Log Target embeddings combining all batches first
            self.projector_tensorboard.add_target_embeddings(target_embeddings, target_labels, target_batch)
        
        # #Add graph to tensorboard
        # if batch_idx==0 and  isinstance(self.logger, TensorBoardLogger):
        #     self.logger.experiment.add_graph(self, (target_batch))
            
    def on_test_end(self):
        labels = self.trainer.test_dataloaders[0].dataset.datasets[0].classes
        #Log Confusion Matrix to TensorBoard
        self.source_confusion_matrix_tensorboard.log_confusion_matrix(labels, "Source")
        self.target_confusion_matrix_tensorboard.log_confusion_matrix(labels, "Target")
        if self.num_sample_test_embeddings_projector > 0. :
            # #Log embeddings to TensorBoard Projector
            self.projector_tensorboard.log_embeddings(labels, num_samples=self.num_sample_test_embeddings_projector)#percentage indicates what percentage of data is loaded to tensorboard
                    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optim.opt, params=[{'params': self.ad_net.parameters()}, {'params': self.backbone.parameters()}, {'params': self.classifier_layer.parameters()}])
        lambda1 = lambda epoch: 1/((1 + self.optim.scheduler.gamma* self.p)**self.optim.scheduler.beta)
        scheduler = {"scheduler": lr_scheduler.LambdaLR(optimizer, lambda1), 'name': 'optimizer_scheduler'}
        return [optimizer], [scheduler]

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        op_out = GradientReversalFn.apply(op_out, coeff)
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        random_out = GradientReversalFn.apply(random_out, coeff)
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    # dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().type_as(ad_out)
    # prepare real and fake label
    source_label = torch.ones(batch_size).type_as(ad_out)
    target_label = torch.zeros(batch_size).type_as(ad_out)
    concat_label = torch.cat((source_label, target_label), 0).long()
    
    if entropy is not None:
        entropy = GradientReversalFn.apply(entropy, coeff)
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.CrossEntropyLoss()(ad_out, concat_label)) / torch.sum(weight).detach().item()
    else:
        try:
            return nn.CrossEntropyLoss()(ad_out, concat_label) 
        except:
            print("Error")