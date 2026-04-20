import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_num_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        bn_res = self.bn(conv_res)
        return self.activation(bn_res)

class UpsampleSkip(nn.Module):

    def __init__(self, skip_channels: int, latent_channels: int, out_channels: int):
        super().__init__()
        self.u1_u2 = nn.Sequential(
            nn.Conv2d(skip_channels, latent_channels, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.u3 = nn.Conv2d(latent_channels, latent_channels, 1, 1, 0)
        self.u4_u5 = nn.Sequential(
            nn.Conv2d(2 * latent_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        u2_res = self.u1_u2(a)
        u3_res = self.u3(b)
        if u2_res.shape[-2:] != u3_res.shape[-2:]:
            u2_res = F.interpolate(u2_res, size=u3_res.shape[-2:], mode='bilinear', align_corners=False)
        u5_res = self.u4_u5(torch.cat([u2_res, u3_res], dim=1))
        return u5_res


class FlowUNet(nn.Module):
    """
    Standard UNet with a few modifications:
     - Uses Bilinear interpolation instead of transposed convolutions
    """

    def __init__(self, input_dim=16, output_dim=2, bev_size=(200,200)) -> None:
        super().__init__()

        self.encoder_step_1 = nn.Sequential(ConvWithNorms(input_dim, input_dim * 2, 3, 2, 1),  # 16->32
                                            ConvWithNorms(input_dim * 2, input_dim * 2, 3, 1, 1),  # 32->32
                                            ConvWithNorms(input_dim * 2, input_dim * 2, 3, 1, 1),  # 32->32
                                            ConvWithNorms(input_dim * 2, input_dim * 2, 3, 1, 1))  # 32->32
        self.encoder_step_2 = nn.Sequential(ConvWithNorms(input_dim * 2, input_dim * 4, 3, 2, 1),  # 32->64
                                            ConvWithNorms(input_dim * 4, input_dim * 4, 3, 1, 1),  # 64->64
                                            ConvWithNorms(input_dim * 4, input_dim * 4, 3, 1, 1),  # 64->64
                                            ConvWithNorms(input_dim * 4, input_dim * 4, 3, 1, 1),  # 64->64
                                            ConvWithNorms(input_dim * 4, input_dim * 4, 3, 1, 1),  # 64->64
                                            ConvWithNorms(input_dim * 4, input_dim * 4, 3, 1, 1))  # 64->64
        self.encoder_step_3 = nn.Sequential(ConvWithNorms(input_dim * 4, input_dim * 8, 3, 2, 1),  # 64->128
                                            ConvWithNorms(input_dim * 8, input_dim * 8, 3, 1, 1),  # 128->128
                                            ConvWithNorms(input_dim * 8, input_dim * 8, 3, 1, 1),  # 128->128
                                            ConvWithNorms(input_dim * 8, input_dim * 8, 3, 1, 1),  # 128->128
                                            ConvWithNorms(input_dim * 8, input_dim * 8, 3, 1, 1),  # 128->128
                                            ConvWithNorms(input_dim * 8, input_dim * 8, 3, 1, 1))  # 128->128

        self.decoder_step1 = UpsampleSkip(input_dim * 8, input_dim * 4, input_dim * 4)
        self.decoder_step2 = UpsampleSkip(input_dim * 4, input_dim * 2, input_dim * 2)
        self.decoder_step3 = UpsampleSkip(input_dim * 2, input_dim, input_dim)
        self.decoder_step4 = nn.Conv2d(input_dim, output_dim, 3, 1, 1)

        self.bev_size = bev_size

    def forward(self, pc0_B: torch.Tensor) -> torch.Tensor:
        bs = pc0_B.shape[0]
        pc0_B = pc0_B.permute(0,2,1).view(bs, -1, self.bev_size[0], self.bev_size[1])
        pc0_F = self.encoder_step_1(pc0_B)
        pc0_L = self.encoder_step_2(pc0_F)
        pc0_R = self.encoder_step_3(pc0_L)

        S = self.decoder_step1(pc0_R, pc0_L)
        T = self.decoder_step2(S, pc0_F)
        U = self.decoder_step3(T, pc0_B)

        V = self.decoder_step4(U)

        V = V.permute(0, 2, 3, 1).view(bs, self.bev_size[0]*self.bev_size[1], -1)
        return V
