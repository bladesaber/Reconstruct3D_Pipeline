import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from typing import Tuple
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        """Computes the structural similarity (SSIM) index map between two images

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:
        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d

class MSGMSLoss(nn.Module):
    def __init__(self, num_scales: int = 3, in_channels: int = 3) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.prewitt_x, self.prewitt_y = self._create_prewitt_kernel()

    def forward(self, img1: Tensor, img2: Tensor, as_loss: bool = True) -> Tensor:
        if not self.prewitt_x.is_cuda or not self.prewitt_y.is_cuda:
            self.prewitt_x = self.prewitt_x.to(img1.device)
            self.prewitt_y = self.prewitt_y.to(img1.device)

        b, c, h, w = img1.shape
        msgms_map = 0
        for scale in range(self.num_scales):

            if scale > 0:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)

            gms_map = self._gms(img1, img2)
            msgms_map += F.interpolate(gms_map, size=(h, w), mode="bilinear", align_corners=False)

        if as_loss:
            return torch.mean(1 - msgms_map / self.num_scales)
        else:
            return torch.mean(1 - msgms_map / self.num_scales, axis=1).unsqueeze(1)

    def _gms(self, img1: Tensor, img2: Tensor) -> Tensor:
        gm1_x = F.conv2d(img1, self.prewitt_x, stride=1, padding=1, groups=self.in_channels)
        gm1_y = F.conv2d(img1, self.prewitt_y, stride=1, padding=1, groups=self.in_channels)
        gm1 = torch.sqrt(gm1_x ** 2 + gm1_y ** 2 + 1e-12)

        gm2_x = F.conv2d(img2, self.prewitt_x, stride=1, padding=1, groups=self.in_channels)
        gm2_y = F.conv2d(img2, self.prewitt_y, stride=1, padding=1, groups=self.in_channels)
        gm2 = torch.sqrt(gm2_x ** 2 + gm2_y ** 2 + 1e-12)

        # Constant c from the following paper. https://arxiv.org/pdf/1308.3052.pdf
        c = 0.0026
        numerator = 2 * gm1 * gm2 + c
        denominator = gm1 ** 2 + gm2 ** 2 + c
        return numerator / (denominator + 1e-12)

    def _create_prewitt_kernel(self) -> Tuple[Tensor, Tensor]:
        prewitt_x = torch.Tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]) / 3.0  # (1, 1, 3, 3)
        prewitt_x = prewitt_x.repeat(self.in_channels, 1, 1, 1)  # (self.in_channels, 1, 3, 3)
        prewitt_y = torch.Tensor([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]]) / 3.0  # (1, 1, 3, 3)
        prewitt_y = prewitt_y.repeat(self.in_channels, 1, 1, 1)  # (self.in_channels, 1, 3, 3)
        return (prewitt_x, prewitt_y)

class Resnet_Perceptual(nn.Module):
    def __init__(self,
                 weights={'layer1': 0.4, 'layer2': 0.35, 'layer3': 0.25},
                 device='cpu'
                 ):
        super(Resnet_Perceptual, self).__init__()

        # self.resnet = resnet50(pretrained=True)
        # train_nodes, eval_nodes = get_graph_node_names(self.resnet)
        self.resnet_return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            # 'layer4': 'layer4',
        }
        self.resnet = create_feature_extractor(
            model=resnet50(pretrained=True),
            return_nodes=self.resnet_return_nodes
        )
        if device == 'cuda':
            self.resnet.to(torch.device('cuda:0'))

        self.weights = weights
        self.init_weight()

    def __call__(self, orig_img, reconst_img, layer_masks):
        orig_rdict = self.resnet(orig_img)
        reconst_rdict = self.resnet(reconst_img)

        content_loss = 0.0
        for key in orig_rdict.keys():
            orig_map = orig_rdict[key]
            reconst_map = reconst_rdict[key]
            layer_mask = layer_masks[key]

            # print(orig_map.shape, layer_mask.shape)
            assert orig_map.shape[-2:] == reconst_map.shape[-2:]
            c_num = orig_map.shape[1]

            loss = (orig_map - reconst_map).abs()
            loss = loss * layer_mask

            loss = loss.sum() / layer_mask.sum() / c_num
            content_loss += loss

        # content_loss = content_loss / float(len(layer_masks))

        return content_loss

    def init_weight(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

if __name__ == '__main__':
    # loss_fn = MSGMSLoss()
    # loss_fn = SSIMLoss()
    # loss_fn = Resnet_Perceptual()
    #
    # print(len(list(loss_fn.named_parameters())))
    # for name, param in loss_fn.named_parameters():
    #     print(name, param.requires_grad)

    pass