import torch.nn as nn

class ClassifierNetwork(nn.Module):
  def __init__(self, out_features, source_classifier_width, n_class, dropout):
    super(ClassifierNetwork, self).__init__()
    
    self.ad_layer1 = nn.Linear(out_features, source_classifier_width)
    self.ad_layer2 = nn.Linear(source_classifier_width, source_classifier_width)
    self.ad_layer3 = nn.Linear(source_classifier_width, n_class)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    if dropout:
      self.dropout1 = nn.Dropout(0.5)
      self.dropout2 = nn.Dropout(0.5)
    self.dropout = dropout
    self.apply(init_weights)
    
  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    if self.dropout:
      x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    if self.dropout:
      x = self.dropout2(x)
    y = self.ad_layer3(x)
    return y
  

def init_weights(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
      nn.init.xavier_normal_(m.weight)
      nn.init.zeros_(m.bias)