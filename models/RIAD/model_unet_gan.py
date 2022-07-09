import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.RIAD.loss_utils import SSIMLoss, MSGMSLoss
from torchmetrics.functional import accuracy

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

class UNet_Gan(nn.Module):
    def __init__(
            self, is_training=True,
            n_channels=3, merge_mode="concat",
            # up_mode="transpose",
            up_mode='bilinear',
    ):
        super(UNet_Gan, self).__init__()

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
        )

        if is_training:
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

            mb_inpaint = self.forward(mask_img)
            mask = mask.unsqueeze(1)
            mb_reconst += mb_inpaint * (1 - mask)

        ### mse loss
        dif = torch.pow((mb_reconst - imgs) * 255. * obj_masks, 2)
        dif = torch.sqrt(torch.sum(dif, dim=(1, 2, 3))) / torch.sum(obj_masks, dim=(1, 2, 3))
        mse_loss = dif.mean()

        ssim_loss = self.ssim_loss_fn(mb_reconst, imgs, as_loss=True)
        msgm_loss = self.msgm_loss_fn(mb_reconst, imgs, as_loss=True)

        return {
            'mse': mse_loss,
            'ssim': ssim_loss,
            'msgm': msgm_loss,
        }, mb_reconst

class Discriminator(nn.Module):
    def __init__(self, width, height):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 32, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 96, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(96, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flattern = nn.Flatten()
        input_c = int(width/32 * height/32)
        self.linear = nn.Linear(input_c, out_features=2, bias=False)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.flattern(feat)
        out = self.linear(feat)
        return out

class CustomTrainer(object):
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, net_d, real_imgs, fake_imgs):
        batch_size = real_imgs.shape[0]

        fake_imgs_detach = fake_imgs.detach()

        g_fake = net_d(fake_imgs)

        d_fake = net_d(fake_imgs_detach)
        d_real = net_d(real_imgs)

        dis_loss = self.loss_fn(d_fake, torch.ones(batch_size, dtype=torch.long)) + \
                   self.loss_fn(d_real, torch.zeros(batch_size, dtype=torch.long))
        gen_loss = self.loss_fn(g_fake, torch.zeros(batch_size, dtype=torch.long))

        d_fake_softmax = F.softmax(d_fake, dim=1)
        d_real_softmax = F.softmax(d_real, dim=1)
        g_fake_softmax = F.softmax(g_fake, dim=1)
        dis_fake_acc = accuracy(d_fake_softmax, torch.ones(batch_size, dtype=torch.long))
        dis_real_acc = accuracy(d_real_softmax, torch.zeros(batch_size, dtype=torch.long))
        dis_acc = (dis_fake_acc + dis_real_acc) / 2.0
        gen_acc = accuracy(g_fake_softmax, torch.zeros(batch_size, dtype=torch.long))

        acc_dict = {
            'dis_acc': dis_acc,
            'gen_acc': gen_acc,
        }
        loss_dict = {
            'dis_loss': dis_loss.mean(),
            'gen_loss': gen_loss.mean(),
        }

        return loss_dict, acc_dict

if __name__ == '__main__':
    from torchsummary import summary

    network = Discriminator(width=640, height=480)
    summary(network, input_size=(3, 640, 480), batch_size=1)
