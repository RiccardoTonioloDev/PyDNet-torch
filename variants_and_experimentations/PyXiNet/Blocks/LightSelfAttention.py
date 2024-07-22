import torch
from torch import nn


class LightSelfAttentionCNN1(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int):
        super(LightSelfAttentionCNN1, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pre_norm = nn.LayerNorm([in_channels, height // 2, width // 2])
        self.post_norm = nn.LayerNorm([in_channels, height // 2, width // 2])

    def forward(self, x):
        batch_size, C, width, height = x.size()
        width, height = width // 2, height // 2
        x = nn.functional.interpolate(x, size=(width, height), mode="area")
        x = self.pre_norm(x)

        # Convoluzioni per ottenere query, key e value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        # Calcolo dell'attenzione
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)

        # Applicazione dell'attenzione ai valori
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Connessione residua
        out = self.gamma * out + x
        out = self.post_norm(out)
        out = nn.functional.interpolate(out, size=(width * 2, height * 2), mode="area")
        return out


class LightSelfAttentionCNN2(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int):
        super(LightSelfAttentionCNN2, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pre_norm = nn.LayerNorm([in_channels, height // 2, width // 2])

    def forward(self, x):
        batch_size, C, width, height = x.size()
        width, height = width // 2, height // 2
        ds_x = nn.functional.interpolate(x, size=(width, height), mode="area")
        ds_x = self.pre_norm(ds_x)

        # Convoluzioni per ottenere query, key e value
        query = (
            self.query_conv(ds_x).view(batch_size, -1, width * height).permute(0, 2, 1)
        )
        key = self.key_conv(ds_x).view(batch_size, -1, width * height)
        value = self.value_conv(ds_x).view(batch_size, -1, width * height)

        # Calcolo dell'attenzione
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)

        # Applicazione dell'attenzione ai valori
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Connessione residua
        gamma_out = self.gamma * out
        gamma_out = nn.functional.interpolate(
            gamma_out, size=(width * 2, height * 2), mode="area"
        )
        out = gamma_out + x
        return out
