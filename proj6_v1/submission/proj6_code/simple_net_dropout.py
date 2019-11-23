import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    '''
    super(SimpleNetDropout, self).__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    self.cnn_layers = nn.Sequential(
        nn.Conv2d(1, 10, 5),
        nn.MaxPool2d(3,3),
        nn.ReLU(),
        nn.Conv2d(10, 20, 5),
        nn.MaxPool2d(3,3),
        nn.ReLU(),
        nn.Dropout(),
    )

    self.fc_layers = nn.Sequential(
        nn.Linear(500, 100),
        nn.Linear(100, 15),
    )

    self.fc_layers = nn.Sequential(
        nn.Linear(500, 100),
        nn.Linear(100, 15),
    )


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
