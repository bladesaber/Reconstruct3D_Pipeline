import os
import shutil
import torch
from torchvision.models import resnet50, resnet18
from torchvision.models import vgg19
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import numpy as np
from torchsummary import summary

# m = resnet18()
# train_nodes, eval_nodes = get_graph_node_names(m)

m = vgg19(pretrained=False)
summary(m, input_size=[(3, 480, 640)], batch_size=1, device="cpu")

# return_nodes = {
#     'layer1': 'layer1',
#     'layer2': 'layer2',
#     'layer3': 'layer3',
#     'layer4': 'layer4',
# }
# m = create_feature_extractor(resnet18(pretrained=True), return_nodes=return_nodes)

# a = torch.from_numpy(
#     np.random.random((1, 3, 96, 96)).astype(np.float32)
# )
# c = m(a)
# for k in c.keys():
#     print(c[k].shape)
