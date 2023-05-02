import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, weight_norm
import torch.nn.functional as F


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 16, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(16, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 64, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 128, (kernel_size, 1), 1, padding=(2, 0))),
            norm_f(nn.Conv2d(128, 128, (3, 1), 1, padding=(1, 0))),
        ])

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.linears = nn.ModuleList([
            norm_f(nn.Linear(128, 128)),
            norm_f(nn.Linear(128, 32)),
        ])

        self.last_linear = norm_f(nn.Linear(32, 1))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        # x = torch.flatten(x, 1, -1)
        x = self.gap(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        fmap.append(x)

        for l in self.linears:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.last_linear(x)
        # x = torch.sigmoid(x)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, x):
        outputs = []
        feature_maps = []
        for i, d in enumerate(self.discriminators):
            output, feature_map = d(x)
            outputs.append(output)
            feature_maps.append(feature_map)

        return outputs, feature_maps


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 16, 41, 2, groups=2, padding=20)),
            norm_f(nn.Conv1d(16, 32, 41, 2, groups=2, padding=20)),
            norm_f(nn.Conv1d(32, 64, 41, 4, groups=2, padding=20)),
            norm_f(nn.Conv1d(64, 128, 41, 4, groups=2, padding=20)),
            norm_f(nn.Conv1d(128, 128, 41, 1, groups=2, padding=20)),
            norm_f(nn.Conv1d(128, 128, 5, 1, padding=2)),
            norm_f(nn.Conv1d(128, 128, 3, 1, padding=1))
        ])

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.linears = nn.ModuleList([
            norm_f(nn.Linear(128, 128)),
            norm_f(nn.Linear(128, 32))
        ])

        self.last_linear = norm_f(nn.Linear(32, 1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        # x = torch.flatten(x, 1, -1)
        x = self.gap(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        fmap.append(x)

        for l in self.linears:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.last_linear(x)
        # x = torch.sigmoid(x)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(2, 1, padding=0),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x):
        outputs = []
        feature_maps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i-1](x)
            output, feature_map = d(x)
            outputs.append(output)
            feature_maps.append(feature_map)

        return outputs, feature_maps


class MetricDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 15, 5)),
            spectral_norm(nn.Conv2d(15, 25, 7)),
            spectral_norm(nn.Conv2d(25, 40, 9)),
            spectral_norm(nn.Conv2d(40, 50, 11))
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_layers = nn.Sequential(
            spectral_norm(nn.Linear(50, 50)),
            spectral_norm(nn.Linear(50, 10)),
        )

        self.last_linear = spectral_norm(nn.Linear(10, 1))

    def forward(self, x):
        for l in self.conv_layers:
            x = l(x)
            x = F.leaky_relu(x, 0.3)

        x = self.gap(x)  # [batch_size, channels, 1, 1]
        x = x.reshape((x.shape[0], x.shape[1]))  # [batch_size, channels]

        for l in self.linear_layers:
            x = l(x)
            x = F.leaky_relu(x, 0.3)

        x = self.last_linear(x)
        # x = torch.sigmoid(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ipt_spect = torch.randn((4, 2, 160, 160)).to(device=device)
    ipt_audio = torch.randn((4, 1, 159 * 159)).to(device=device)

    md = MetricDiscriminator().to(device=device)
    opt = md(ipt_spect)
    print(opt.shape)

    msd = MultiScaleDiscriminator().to(device=device)
    mpd = MultiPeriodDiscriminator().to(device=device)

    msd_outputs, msd_feature_maps = msd(ipt_audio)
    mpd_outputs, mpd_feature_maps = mpd(ipt_audio)

    print(msd_outputs[1].shape)
    print(msd_feature_maps[1][-1].shape)
    print(mpd_outputs[1].shape)
    print(mpd_feature_maps[1][-1].shape)

