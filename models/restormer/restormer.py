import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.restormer.loss_utils import l1_loss
from models.restormer.loss_utils import mse_loss

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm_WithBias(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm_WithBias, self).__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        x = to_3d(x)

        sigma = x.var(-1, keepdim=True, unbiased=False)
        mu = x.mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

        return to_4d(x, h, w)

class LayerNorm_WithoutBias(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm_WithoutBias, self).__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        x = to_3d(x)

        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

        return to_4d(x, h, w)

class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True):
        super(GDFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        ### 这里是通道分组卷积
        self.dwconv = nn.Conv2d(
            hidden_features*2, hidden_features*2, kernel_size=3,
            stride=1, padding=1, groups=hidden_features*2, bias=bias
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=3, bias=bias)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.relu(x)
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        ### todo 不使用共享参数会不会泛化更好 ??
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)

        ### todo 这里是通道分组卷积,减少了大量参数，但对效果有没抑制??
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        ### channel softmax
        attn = attn.softmax(dim=-1)

        # out = (attn @ v)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm_WithBias(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm_WithBias(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class PixelShuttle_UpSample(nn.Module):
    def __init__(self, n_feat):
        super(PixelShuttle_UpSample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class PixelShuttle_Downsample(nn.Module):
    def __init__(self, n_feat):
        super(PixelShuttle_Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Interpolation_UpSample(nn.Module):
    def __init__(self):
        super(Interpolation_UpSample, self).__init__()
        # self.net = nn.Upsample(scale_factor=2)
        # self.net = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return F.interpolate(x, scale_factor=2)

class Interpolation_Downsample(nn.Module):
    def __init__(self):
        super(Interpolation_Downsample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=0.5)

class Restormer(nn.Module):
    '''
    todo
    可以考虑结合传统预训练网络 resnet
    '''
    def __init__(self,
        input_dim=3,
        out_dim=3,
        embed_dim = 48,
        num_blocks = (4,6,6,8),
        # num_refinement_blocks = 4,
        heads = (1,2,4,8),
        ffn_expansion_factor = 2.66,
        bias = False,
        # LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(Restormer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = out_dim
        self.embed_dim = embed_dim

        self.dual_pixel_task = dual_pixel_task
        self.num_blocks = num_blocks
        self.heads = heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias

        self.init_network()
        self.pixel_loss_weight = 1.0

    def init_network(self):
        self.patch_embed = nn.Conv2d(
            self.input_dim, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=self.bias
        )

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=self.embed_dim, num_heads=self.heads[0],
                ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias
            ) for i in range(self.num_blocks[0])
        ])

        self.down1_2 = PixelShuttle_Downsample(self.embed_dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=self.embed_dim*2, num_heads=self.heads[1],
                ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias
            ) for i in range(self.num_blocks[1])
        ])

        self.down2_3 = PixelShuttle_Downsample(self.embed_dim*2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=self.embed_dim*4, num_heads=self.heads[2],
                ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias
            ) for i in range(self.num_blocks[2])
        ])

        self.down3_4 = PixelShuttle_Downsample(self.embed_dim*4)
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=self.embed_dim*8, num_heads=self.heads[3],
                ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias
            ) for i in range(self.num_blocks[3])
        ])

        self.up4_3 = PixelShuttle_UpSample(self.embed_dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(self.embed_dim*8, self.embed_dim*4, kernel_size=1, bias=self.bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=self.embed_dim*4, num_heads=self.heads[2],
                ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias
            ) for i in range(self.num_blocks[2])
        ])

        self.up3_2 = PixelShuttle_UpSample(self.embed_dim*4)
        self.reduce_chan_level2 = nn.Conv2d(self.embed_dim*4, self.embed_dim*2, kernel_size=1, bias=self.bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=self.embed_dim*2, num_heads=self.heads[1],
                ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias
            ) for i in range(self.num_blocks[1])
        ])

        self.up2_1 = PixelShuttle_UpSample(self.embed_dim*2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=self.embed_dim*2, num_heads=self.heads[0],
                ffn_expansion_factor=self.ffn_expansion_factor, bias=self.bias
            ) for i in range(self.num_blocks[0])
        ])

        # self.refinement = nn.Sequential(*[
        #     TransformerBlock(dim=embed_dim*2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        #     for i in range(num_refinement_blocks)
        # ])

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(self.embed_dim, self.embed_dim*2, kernel_size=1, bias=self.bias)

        self.output = nn.Conv2d(self.embed_dim*2, self.output_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)

        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

    def train_step(self, noise_img, gt_img):
        reconstruct_img = self(noise_img)
        pixel_loss = l1_loss(reconstruct_img, gt_img)
        return pixel_loss, reconstruct_img