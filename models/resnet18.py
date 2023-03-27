import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.model = resnet18(weights=None)
        self.model.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )

    def forward(self, t):
        return self.model.forward(t)
    
    def predict(self, x):
        return self.forward(x).argmax(dim=1)

    def predict_proba(self, x):
        return self.forward(x)

    def add_dropout(self, dropout):
        def append_dropout(model, rate=0.2):
            for name, module in model.named_children():
                if len(list(module.children())) > 0:
                    append_dropout(module)
                if isinstance(module, nn.ReLU):
                    new = nn.Sequential(module, nn.Dropout(p=rate, inplace=False))
                    setattr(model, name, new)        
        append_dropout(self.model, rate=dropout)
