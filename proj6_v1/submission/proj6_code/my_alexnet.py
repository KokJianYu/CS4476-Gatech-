import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super(MyAlexNet, self).__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    model=alexnet(pretrained=True)
    self.cnn_layers = nn.Sequential(
        *list(model.children())[0],
        list(model.children())[1],
    )
    self.fc_layers = nn.Sequential(
        *list(model.children())[2][:-1],
        nn.Linear(4096, 15)
    )
    # freezing the layers by setting requires_grad=False
    # example: self.cnn_layers[idx].weight.requires_grad = False
    for i in range(len(self.cnn_layers)):
        if hasattr(self.cnn_layers[i], 'weight'):
            self.cnn_layers[i].weight.requires_grad = False
        if hasattr(self.cnn_layers[i], 'bias'):            
            self.cnn_layers[i].bias.requires_grad = False

    for i in range(len(self.fc_layers)-1):
        if hasattr(self.fc_layers[i], 'weight'):
            self.fc_layers[i].weight.requires_grad = False
        if hasattr(self.fc_layers[i], 'bias'):  
            self.fc_layers[i].bias.requires_grad = False


    # take care to turn off gradients for both weight and bias


    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''

    model_output = None
    x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

    ###########################################################################
    # Student code begin
    ###########################################################################
    cnn_output = self.cnn_layers(x)
    flatten_output = cnn_output.view(x.shape[0],-1)
    model_output = self.fc_layers(flatten_output)

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
