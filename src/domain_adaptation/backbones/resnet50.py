from torch import nn
import torchvision
import pytorch_lightning as pl

class Resnet50(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.out_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        

    def forward(self, x):
        embeddings = self.model(x)
        return embeddings
    
    def out_dimension(self):
        return self.out_features