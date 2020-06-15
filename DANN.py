import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.autograd import Function
#from gradient_reversal import ReverseLayerF

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None




class DANNModel(nn.Module):

    def __init__(self, num_classes=1000):
        super(DANNModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.discriminator = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x, alpha = None):
    #if we pass alpha we can assume we are training the discriminator
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if alpha is None:
          class_out = self.classifier(x)
          return class_out
        
        else:
          #gradient reversal
          reverse_features = ReverseLayerF.apply(x, alpha)
          discr_out = self.discriminator(reverse_features)
          return discr_out
            
          
          
          
         
          

def alexnetDANN(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = DANNModel(num_classes = 1000, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict = False)
     
    #we copy the pretrained weights of the classifier to the domain classifier
    model.discriminator[1].weight.data = model.classifier[1].weight.data.clone()
    model.discriminator[1].bias.data = model.classifier[1].bias.data.clone()

    model.discriminator[4].weight.data = model.classifier[4].weight.data.clone()
    model.discriminator[4].bias.data = model.classifier[4].bias.data.clone()

    return model
