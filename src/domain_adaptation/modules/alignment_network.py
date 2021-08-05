import torch.nn as nn

class AlignmentNetwork(nn.Module):
  def __init__(self, out_features, alignement_width):
    super(AlignmentNetwork, self).__init__()
    self.alignmnet_layer = nn.Sequential(nn.Linear(out_features, alignement_width), nn.BatchNorm1d(alignement_width), nn.ReLU(), nn.Dropout(0.5))
    
    self.alignmnet_layer[0].weight.data.normal_(0, 0.005)
    self.alignmnet_layer[0].bias.data.fill_(0.1)

  def forward(self, x):
    y = self.alignmnet_layer(x)
    return y