import torch.nn as nn
from typing import Iterable
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchmetrics.functional import accuracy

class Resnet50_test(nn.Module):
    def __init__(self, layers: Iterable[str]):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}
        self.out_dims = []

        for layer_id in layers:
            layer = dict([*self.backbone.named_modules()])[layer_id]
            layer.register_forward_hook(self.get_features(layer_id))
            # get output dimension of features if available
            layer_modules = [*layer.modules()]
            for idx in reversed(range(len(layer_modules))):
                if hasattr(layer_modules[idx], "out_channels"):
                    self.out_dims.append(layer_modules[idx].out_channels)
                    break

    def get_features(self, layer_id: str):
        def hook(_, __, output):
            self._features[layer_id] = output
        return hook

    def forward(self, input_tensor):
        self._features = {layer: torch.empty(0) for layer in self.layers}
        _ = self.backbone(input_tensor)
        return self._features

class Resnet18_model(nn.Module):
    def __init__(self, is_train=True, with_init=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 32, bias=True)
        self.out_relu = nn.ReLU()
        self.out_linear = nn.Linear(32, 2, bias=False)

        if is_train:
            self.loss_fn = nn.CrossEntropyLoss()

        if with_init:
            self.init_weight()

    def forward(self, x):
        x = self.backbone(x)
        x = self.out_relu(x)
        x = self.out_linear(x)

        if not self.training:
            x = F.softmax(x, dim=1)

        return x

    def train_step(self, x, labels):
        x = self.forward(x)
        loss = self.loss_fn(x, labels)

        out = F.softmax(x, dim=1)
        acc = accuracy(out, labels)

        return loss, acc

    def init_weight(self):
        for name, param in self.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            elif 'out_linear' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        torch.nn.init.constant_(self.backbone.fc.bias, 0.0)
        torch.nn.init.normal_(self.backbone.fc.weight)
        torch.nn.init.normal_(self.out_linear.weight)

class Resnet50_model(nn.Module):
    def __init__(self, is_train=True, with_init=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, 64, bias=True)
        self.out_linear = nn.Linear(64, 2, bias=False)

        if is_train:
            self.loss_fn = nn.CrossEntropyLoss()

        if with_init:
            self.init_weight()

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = self.out_linear(x)

        if not self.training:
            x = F.softmax(x, dim=1)

        return x

    def train_step(self, x, labels):
        x = self.forward(x)
        loss = self.loss_fn(x, labels)

        out = F.softmax(x, dim=1)
        acc = accuracy(out, labels)

        return loss, acc

    def init_weight(self):
        for name, param in self.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            elif 'out_linear' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        torch.nn.init.constant_(self.backbone.fc.bias, 0.0)
        torch.nn.init.normal_(self.backbone.fc.weight)
        torch.nn.init.normal_(self.out_linear.weight)

if __name__ == '__main__':
    net = Resnet18_model()
    for name, param in net.named_parameters():
        print(name)
        # print(param)
        print('-----------------')
