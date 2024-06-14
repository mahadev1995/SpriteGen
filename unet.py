import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_residual=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch//2, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, out_ch//2)
        self.activation1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch//2, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.use_residual = False

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.use_residual:
            out = F.gelu(out + x)
        return out


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

    
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

        self.time_emb = nn.Sequential(nn.SiLU(),
                                      nn.Linear(
                                          emb_dim,
                                          out_ch
                                      ),
                                      )
    
    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.conv(x)
        emb = self.time_emb(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x+emb
    

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)
        self.time_emb = nn.Sequential(nn.SiLU(),
                                      nn.Linear(
                                          emb_dim,
                                          out_ch
                                      ),
                                      )
        
    def forward(self, x, enc_x, t):
        x = self.upsample(x)
        x = torch.cat([enc_x, x], dim=1)
        x = self.conv(x)
        emb = self.time_emb(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x+emb
    

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, time_dim=256, device='cuda'):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.conv = ConvBlock(in_ch, 64)
        self.down1 = Down(64, 128)
        self.attn1 = SelfAttention(128, 8)
        self.down2 = Down(128, 256)
        self.attn2 = SelfAttention(256, 4)

        self.bot1 = ConvBlock(256, 512)
        self.bot2 = ConvBlock(512, 512)
        self.bot3 = ConvBlock(512, 256)

        self.up1 = Up(384, 128)
        self.attn3 = SelfAttention(128, 8)
        self.up2 = Up(192, 64)
        self.attn4 = SelfAttention(64, 16)

        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def sinusoidal_embedding(self, t, ch):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, ch, 2, device=self.device).float() / ch)
        )
        emb_a = torch.sin(t.repeat(1, ch//2)*inv_freq)
        emb_b = torch.cos(t.repeat(1, ch//2)*inv_freq)
        emb = torch.cat([emb_a, emb_b], dim=-1)
        return emb
    
    def forward(self, x, t):
        t = t[..., None].type(torch.float)
        t = self.sinusoidal_embedding(t, self.time_dim)
        
        x1 = self.conv(x)
        x2 = self.down1(x1, t)
        x2 = self.attn1(x2)
        x3 = self.down2(x2, t)
        x3 = self.attn2(x3)

        x3 = self.bot1(x3)
        x3 = self.bot2(x3)
        x3 = self.bot3(x3)

        x = self.up1(x3, x2, t)
        x = self.attn3(x)
        x = self.up2(x, x1, t)
        x = self.attn4(x)

        out = self.out_conv(x)
        return out


        


