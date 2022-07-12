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
from torchvision.datasets.flowers102 import Flowers102
from torchvision.datasets.caltech import Caltech101


