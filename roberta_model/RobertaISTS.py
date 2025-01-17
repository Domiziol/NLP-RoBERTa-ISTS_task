import torch
import torch.nn as nn
from torch import Tensor
from yacs.config import CfgNode
import torch.nn.functional as functional

# DO NOT DELETE, even though these imports are unused without them the program crashes
import torch.optim as optim
from fairseq.models.roberta import RobertaModel


class RobertaISTS(torch.nn.Module):
    IN_FEATURES = 1024

    def __init__(self, num_classes: int, dropout_rate: float, hidden_neurons: int) -> None:
        super(RobertaISTS, self).__init__()

        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.linear1 = nn.Linear(in_features=self.IN_FEATURES, out_features=hidden_neurons, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_neurons, out_features=1, bias=True)
        self.linear3 = nn.Linear(in_features=self.IN_FEATURES, out_features=hidden_neurons, bias=True)
        self.linear4 = nn.Linear(in_features=hidden_neurons, out_features=num_classes, bias=True)
        self.dropout1 = nn.Dropout(p=dropout_rate, inplace=False)
        self.dropout2 = nn.Dropout(p=dropout_rate, inplace=False)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # not sure whether the first val of the tuple is a Tensor
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        features = self.roberta.extract_features(x)
        x = torch.mean(features, 1, keepdim=False)
        x = self.dropout1(x)

        # STS value prediction
        x1 = self.relu1(self.linear1(x))
        x1 = self.dropout2(x1)
        out1 = self.linear2(x1)[0]

        # explanatory layer
        x2 = self.relu2(self.linear3(x))
        out2 = functional.log_softmax(self.linear4(x2), dim=1)

        return out1, out2

    @classmethod
    def build_model_from_cfg(cls, config: CfgNode) -> "RobertaISTS":
        return RobertaISTS(config.MODEL.NUM_CLASSES,
                           config.MODEL.DROPOUT,
                           config.MODEL.HIDDEN_NEURONS)
