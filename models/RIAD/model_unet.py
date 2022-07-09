import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from models.RIAD.loss_utils import SSIMLoss, MSGMSLoss
from models.RIAD.loss_utils import Resnet_Perceptual

def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1, stride=1, padding=0
            ),
        )

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3, stride=1, padding=self.padding
        )
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode="concat", up_mode="transpose"):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == "concat":
            self.conv1 = nn.Conv2d(
                in_channels=2 * self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3, stride=1, padding=1
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3, stride=1, padding=1
            )
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu2 = nn.ReLU()

    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)

        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x

class UNet(nn.Module):
    def __init__(
            self, is_training=True,
            n_channels=3, merge_mode="concat",
            # up_mode="transpose",
            up_mode='bilinear',
    ):
        super(UNet, self).__init__()

        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlock(self.n_chnnels, 64, 3, 1, 1)
        self.down2 = UNetDownBlock(64, 128, 4, 2, 1)
        self.down3 = UNetDownBlock(128, 256, 4, 2, 1)
        self.down4 = UNetDownBlock(256, 512, 4, 2, 1)
        self.down5 = UNetDownBlock(512, 512, 4, 2, 1)

        self.up1 = UNetUpBlock(512, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)

        self.conv_final = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=3,
                kernel_size=3, stride=1, padding=1
            ),

            nn.Tanh()
            # nn.Sigmoid()
        )

        if is_training:
            # self.mse_loss_fn = nn.MSELoss()
            self.ssim_loss_fn = SSIMLoss(kernel_size=11, sigma=0.5)
            self.msgm_loss_fn = MSGMSLoss(num_scales=3)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)

        return x

    def infer(self, disjoint_masks, mask_imgs):
        mb_reconst = 0
        mask_num = disjoint_masks.shape[1]

        for mask_id in range(mask_num):
            mask_img = mask_imgs[:, mask_id, :, :, :]
            mask = disjoint_masks[:, mask_id, :, :]

            mb_inpaint = self.forward(mask_img)
            mask = mask.unsqueeze(1)
            mb_reconst += mb_inpaint * (1 - mask)

        return mb_reconst

    def train_step(self, imgs, disjoint_masks, mask_imgs, obj_masks):
        mb_reconst = 0
        mask_num = disjoint_masks.shape[1]

        for mask_id in range(mask_num):
            mask_img = mask_imgs[:, mask_id, :, :, :]
            mask = disjoint_masks[:, mask_id, :, :]

            # temp_mask_img = mask_img.cpu().numpy()
            # temp_mask = mask.cpu().numpy()
            # for temp_id in range(temp_mask_img.shape[0]):
            #     temp_img = temp_mask_img[temp_id, ...]
            #     temp_mask_single = temp_mask[temp_id, ...]
            #     temp_img = np.transpose(temp_img, (1, 2, 0))
            #     plt.figure('1')
            #     plt.imshow(temp_img)
            #     plt.figure('2')
            #     plt.imshow(temp_mask_single)
            #     plt.show()

            mb_inpaint = self.forward(mask_img)
            mask = mask.unsqueeze(1)
            mb_reconst += mb_inpaint * (1 - mask)

        # mb_reconst = mb_reconst / mask_num
        # temp = mb_reconst.detach().cpu().numpy()
        # print(temp.max(), temp.min())

        # mse_loss = self.mse_loss_fn(mb_reconst, imgs)
        dif = torch.pow((mb_reconst - imgs) * 255. * obj_masks, 2)
        dif = torch.sqrt(torch.sum(dif, dim=(1, 2, 3))) / torch.sum(obj_masks, dim=(1, 2, 3))
        mse_loss = dif.mean()

        ssim_loss = self.ssim_loss_fn(mb_reconst, imgs, as_loss=True)
        msgm_loss = self.msgm_loss_fn(mb_reconst, imgs, as_loss=True)
        total_loss = mse_loss + ssim_loss + msgm_loss

        return {
            'mse': mse_loss,
            'ssim': ssim_loss,
            'msgm': msgm_loss,
            'total': total_loss,
            'rimgs': mb_reconst
        }

class UNet_perception(nn.Module):
    def __init__(
            self, is_training=True,
            n_channels=3, merge_mode="concat",
            # up_mode="transpose",
            up_mode='bilinear',
            device='cpu'
    ):
        super(UNet_perception, self).__init__()

        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlock(self.n_chnnels, 64, 3, 1, 1)
        self.down2 = UNetDownBlock(64, 128, 4, 2, 1)
        self.down3 = UNetDownBlock(128, 256, 4, 2, 1)
        self.down4 = UNetDownBlock(256, 512, 4, 2, 1)
        self.down5 = UNetDownBlock(512, 512, 4, 2, 1)

        self.up1 = UNetUpBlock(512, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)

        self.conv_final = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=3,
                kernel_size=3, stride=1, padding=1
            ),

            nn.Tanh()
            # nn.Sigmoid()
        )

        if is_training:
            self.ssim_loss_fn = SSIMLoss(kernel_size=11, sigma=0.5)
            self.msgm_loss_fn = MSGMSLoss(num_scales=3)

            self.perceptual_loss_fn = Resnet_Perceptual(device=device)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)

        return x

    def infer(self, disjoint_masks, mask_imgs):
        mb_reconst = 0
        mask_num = disjoint_masks.shape[1]

        for mask_id in range(mask_num):
            mask_img = mask_imgs[:, mask_id, :, :, :]
            mask = disjoint_masks[:, mask_id, :, :]

            mb_inpaint = self.forward(mask_img)
            mask = mask.unsqueeze(1)
            mb_reconst += mb_inpaint * (1 - mask)

        return mb_reconst

    def train_step(self,
                   imgs, disjoint_masks, disjoint_imgs, obj_masks,
                   layer1_mask, layer2_mask, layer3_mask):
        mb_reconst = 0
        mask_num = disjoint_masks.shape[1]

        for mask_id in range(mask_num):
            mask_img = disjoint_imgs[:, mask_id, :, :, :]
            mask = disjoint_masks[:, mask_id, :, :]

            # temp_mask_img = mask_img.cpu().numpy()
            # temp_mask = mask.cpu().numpy()
            # for temp_id in range(temp_mask_img.shape[0]):
            #     temp_img = temp_mask_img[temp_id, ...]
            #     temp_mask_single = temp_mask[temp_id, ...]
            #     temp_img = np.transpose(temp_img, (1, 2, 0))
            #     plt.figure('1')
            #     plt.imshow(temp_img)
            #     plt.figure('2')
            #     plt.imshow(temp_mask_single)
            #     plt.show()

            mb_inpaint = self.forward(mask_img)
            mask = mask.unsqueeze(1)
            mb_reconst += mb_inpaint * (1 - mask)

        dif = torch.pow((mb_reconst - imgs) * 255. * obj_masks, 2)
        dif = torch.sqrt(torch.sum(dif, dim=(1, 2, 3)))/torch.sum(obj_masks, dim=(1, 2, 3))
        mse_loss = dif.mean()

        ssim_loss = self.ssim_loss_fn(mb_reconst, imgs, as_loss=True)
        msgm_loss = self.msgm_loss_fn(mb_reconst, imgs, as_loss=True)

        layer_mask_dict = {
            'layer1': layer1_mask,
            'layer2': layer2_mask,
            'layer3': layer3_mask,
        }
        perce_loss = self.perceptual_loss_fn(orig_img=imgs, reconst_img=mb_reconst, layer_masks=layer_mask_dict)
        perce_loss = perce_loss * 3.0

        total_loss = mse_loss + ssim_loss + msgm_loss + perce_loss

        return {
            'mse': mse_loss,
            'ssim': ssim_loss,
            'msgm': msgm_loss,
            'perce_loss': perce_loss,
            'total': total_loss,
            'rimgs': mb_reconst
        }

if __name__ == '__main__':
    from torchsummary import summary

    network = UNet()
    summary(network, input_size=(3, 480, 640), batch_size=1)
