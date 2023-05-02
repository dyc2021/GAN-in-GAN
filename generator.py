import ptflops
import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_normalize(input, dim=1):
    """
    Perform complex normalization
    """
    real_value, imag_value = input.real, input.imag
    real_norm = F.normalize(real_value, dim=dim)
    imag_norm = F.normalize(imag_value, dim=dim)

    return real_norm.type(torch.complex64) + 1j * imag_norm.type(torch.complex64)


def complex_softmax(input, dim=1):
    real_value, imag_value = input.real, input.imag
    real_softmax = F.softmax(real_value, dim=dim)
    imag_softmax = F.softmax(imag_value, dim=dim)

    return real_softmax.type(torch.complex64) + 1j * imag_softmax.type(torch.complex64)


def complex_gelu(input):
    real_value, imag_value = input.real, input.imag
    real_gelu = F.gelu(real_value)
    imag_gelu = F.gelu(imag_value)

    return real_gelu.type(torch.complex64) + 1j * imag_gelu.type(torch.complex64)


def complex_matmul(first_matrix, second_matrix):
    """
        Performs the matrix product between two complex matricess
    """
    output_real = torch.matmul(first_matrix.real, second_matrix.real) - \
                  torch.matmul(first_matrix.imag, second_matrix.imag)
    output_imag = torch.matmul(first_matrix.real, second_matrix.imag) + \
                  torch.matmul(first_matrix.imag, second_matrix.real)

    return output_real.type(torch.complex64) + 1j * output_imag.type(torch.complex64)


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) \
           + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape,
                 eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super(ComplexLayerNorm, self).__init__()
        self.layernorm_real = nn.LayerNorm(normalized_shape, eps, elementwise_affine, device, dtype)
        self.layernorm_imag = nn.LayerNorm(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, input):
        real_value, imag_value = input.real, input.imag
        layernorm_real = self.layernorm_real(real_value)
        layernorm_imag = self.layernorm_imag(imag_value)

        return layernorm_real.type(torch.complex64) + 1j * layernorm_imag.type(torch.complex64)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex(self.conv_real, self.conv_imag, input)


class ComplexMDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(ComplexMDTA, self).__init__()
        self.num_heads = num_heads
        self.real_temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.imag_temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = ComplexConv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = ComplexConv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3,
                                      bias=False)
        self.project_out = ComplexConv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = complex_normalize(q, dim=-1), complex_normalize(k, dim=-1)

        pre_attn = complex_matmul(q, k.transpose(-2, -1).contiguous())
        pre_attn_temp_real, pre_attn_temp_imag = pre_attn.real * self.real_temperature, pre_attn.imag * self.imag_temperature
        attn = complex_softmax(torch.complex(pre_attn_temp_real, pre_attn_temp_imag), dim=-1)
        out = self.project_out(complex_matmul(attn, v).reshape(b, -1, h, w))
        return out


class ComplexGDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(ComplexGDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = ComplexConv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = ComplexConv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                                  groups=hidden_channels * 2, bias=False)
        self.project_out = ComplexConv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(complex_gelu(x1) * x2)
        return x


class ComplexChannelAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(ComplexChannelAttentionBlock, self).__init__()

        self.norm1 = ComplexLayerNorm(channels)
        self.attn = ComplexMDTA(channels, num_heads)
        self.norm2 = ComplexLayerNorm(channels)
        self.ffn = ComplexGDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class ComplexDownSample(nn.Module):
    def __init__(self, channels):
        super(ComplexDownSample, self).__init__()
        self.body = nn.Sequential(ComplexConv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class ComplexUpSample(nn.Module):
    def __init__(self, channels):
        super(ComplexUpSample, self).__init__()
        self.body = nn.Sequential(ComplexConv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class ComplexRestormer(nn.Module):
    def __init__(self,
                 # num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384],
                 # num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[24, 48, 96, 192],
                 num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[12, 24, 48, 96],
                 num_refinement=4,
                 # expansion_factor=2.66
                 expansion_factor=1.5,
                 n_fft=318,
                 spectrogram_compress_factor=0.5,
                 tanh_scaling_factor=1.1
                 ):
        super(ComplexRestormer, self).__init__()

        self.spectrogram_compress_factor = spectrogram_compress_factor
        self.tanh_scaling_factor = tanh_scaling_factor

        self.n_fft = n_fft
        self.win_length = self.n_fft
        self.hop_length = self.win_length // 2
        self.window = nn.Parameter(torch.hann_window(window_length=self.win_length), requires_grad=False)

        self.embed_conv = ComplexConv2d(1, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[ComplexChannelAttentionBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([ComplexDownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([ComplexUpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([ComplexConv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList(
            [nn.Sequential(*[ComplexChannelAttentionBlock(channels[2], num_heads[2], expansion_factor)
                             for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = ComplexConv2d(channels[1], 1, kernel_size=3, padding=1, bias=False)

    def forward(self, audio):
        x_complex = torch.stft(audio, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                               window=self.window,
                               return_complex=True).unsqueeze(1)

        x_mag = torch.sqrt(x_complex.real ** 2 + x_complex.imag ** 2)
        x_phase = torch.atan2(x_complex.imag, x_complex.real)

        x_compressed_mag = x_mag ** self.spectrogram_compress_factor

        x_complex_compressed = torch.complex(x_compressed_mag * torch.cos(x_phase),
                                             x_compressed_mag * torch.sin(x_phase))

        fo = self.embed_conv(x_complex_compressed)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out_complex_compressed = self.output(fr)

        out_compressed_mag = torch.sqrt(out_complex_compressed.real ** 2 + out_complex_compressed.imag ** 2)
        out_phase = torch.atan2(out_complex_compressed.imag, out_complex_compressed.real)

        clean_compressed_mag = self.tanh_scaling_factor * torch.tanh(out_compressed_mag) * x_compressed_mag
        clean_phase = out_phase + x_phase

        clean_mag = clean_compressed_mag ** (1.0 / self.spectrogram_compress_factor)

        clean_complex = torch.complex(clean_mag * torch.cos(clean_phase), clean_mag * torch.sin(clean_phase))

        return (x_complex_compressed.squeeze(1),
                out_complex_compressed.squeeze(1),
                torch.istft(clean_complex.squeeze(1),
                            n_fft=self.n_fft,
                            win_length=self.win_length,
                            window=self.window,
                            hop_length=self.hop_length)
                )

    def only_stft_forward(self, audio):
        return torch.stft(audio, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                          window=self.window,
                          return_complex=True)

    def only_istft_forward(self, x_complex):
        return torch.istft(x_complex, n_fft=self.n_fft, win_length=self.win_length,
                           window=self.window,
                           hop_length=self.hop_length)


class SpectroComplexRestormer(nn.Module):
    def __init__(self,
                 # num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384],
                 # num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[24, 48, 96, 192],
                 num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[12, 24, 48, 96],
                 num_refinement=4,
                 # expansion_factor=2.66
                 expansion_factor=1.5,
                 tanh_scaling_factor=1.1
                 ):
        super(SpectroComplexRestormer, self).__init__()

        self.tanh_scaling_factor = tanh_scaling_factor

        self.embed_conv = ComplexConv2d(1, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[ComplexChannelAttentionBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([ComplexDownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([ComplexUpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([ComplexConv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList(
            [nn.Sequential(*[ComplexChannelAttentionBlock(channels[2], num_heads[2], expansion_factor)
                             for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = ComplexConv2d(channels[1], 1, kernel_size=3, padding=1, bias=False)

    def forward(self, spectrogram_complex):
        spectrogram_complex_ = spectrogram_complex.unsqueeze(1)

        spectrogram_phase = torch.atan2(spectrogram_complex_.imag, spectrogram_complex_.real)
        spectrogram_mag = torch.sqrt(spectrogram_complex_.real ** 2 + spectrogram_complex_.imag ** 2)

        fo = self.embed_conv(spectrogram_complex_)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out_complex = self.output(fr)

        out_mag = torch.sqrt(out_complex.real ** 2 + out_complex.imag ** 2)
        out_phase = torch.atan2(out_complex.imag, out_complex.real)

        clean_mag = self.tanh_scaling_factor * torch.tanh(out_mag) * spectrogram_mag
        clean_phase = out_phase + spectrogram_phase

        clean_complex = torch.complex(clean_mag * torch.cos(clean_phase), clean_mag * torch.sin(clean_phase)).squeeze(1)

        return clean_complex


class SpectroComplexDualRestormer(nn.Module):
    def __init__(self,
                 # num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384],
                 # num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[24, 48, 96, 192],
                 num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[12, 24, 48, 96],
                 num_refinement=4,
                 # expansion_factor=2.66
                 expansion_factor=1.5,
                 tanh_scaling_factor=1.0
                 ):
        super(SpectroComplexDualRestormer, self).__init__()

        self.tanh_scaling_factor = tanh_scaling_factor

        self.embed_conv = ComplexConv2d(1, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[ComplexChannelAttentionBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([ComplexDownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([ComplexUpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([ComplexConv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList(
            [nn.Sequential(*[ComplexChannelAttentionBlock(channels[2], num_heads[2], expansion_factor)
                             for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement_mag = nn.Sequential(*[ComplexChannelAttentionBlock(channels[1], num_heads[0], expansion_factor)
                                              for _ in range(num_refinement)])
        self.output_mag = ComplexConv2d(channels[1], 1, kernel_size=3, padding=1, bias=False)

        self.refinement_phase = nn.Sequential(
            *[ComplexChannelAttentionBlock(channels[1], num_heads[0], expansion_factor)
              for _ in range(num_refinement)])
        self.output_phase = ComplexConv2d(channels[1], 1, kernel_size=3, padding=1, bias=False)

    def forward(self, spectrogram_complex):
        spectrogram_complex_ = spectrogram_complex.unsqueeze(1)

        spectrogram_phase = torch.atan2(spectrogram_complex_.imag, spectrogram_complex_.real)
        spectrogram_mag = torch.sqrt(spectrogram_complex_.real ** 2 + spectrogram_complex_.imag ** 2)

        fo = self.embed_conv(spectrogram_complex_)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))

        fr_mag = self.refinement_mag(fd)
        out_mag_mask = self.output_mag(fr_mag)

        fr_phase = self.refinement_phase(fd)
        out_phase_complex = self.output_phase(fr_phase)

        clean_mag = self.tanh_scaling_factor * \
                    torch.tanh(torch.sqrt(out_mag_mask.real ** 2 + out_mag_mask.imag ** 2)) * spectrogram_mag

        clean_complex = torch.complex(clean_mag * torch.cos(spectrogram_phase) + out_phase_complex.real,
                                      clean_mag * torch.sin(spectrogram_phase) + out_phase_complex.imag).squeeze(1)

        return clean_complex


class SpectroDualRestormer(nn.Module):
    def __init__(self,
                 # num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384],
                 # num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[24, 48, 96, 192],
                 num_blocks=[2, 3, 4, 4], num_heads=[1, 2, 4, 8], channels=[12, 24, 48, 96],
                 num_refinement=4,
                 # expansion_factor=2.66
                 expansion_factor=1.5
                 ):
        super(SpectroDualRestormer, self).__init__()

        channels = channels * 2

        self.embed_conv = nn.Conv2d(2, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement_mag = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                              for _ in range(num_refinement)])
        self.output_mag = nn.Conv2d(channels[1], 1, kernel_size=3, padding=1, bias=False)

        self.refinement_phase = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                                for _ in range(num_refinement)])
        self.output_phase = nn.Conv2d(channels[1], 2, kernel_size=3, padding=1, bias=False)

    def forward(self, spectrogram_complex):
        spectrogram_complex_ = spectrogram_complex.unsqueeze(1)

        spectrogram_phase = torch.atan2(spectrogram_complex_.imag, spectrogram_complex_.real)
        spectrogram_mag = torch.sqrt(spectrogram_complex_.real ** 2 + spectrogram_complex_.imag ** 2)

        fo = self.embed_conv(torch.cat((spectrogram_complex_.real, spectrogram_complex_.imag), dim=1))
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))

        fr_mag = self.refinement_mag(fd)
        out_mag_mask = self.output_mag(fr_mag)

        fr_phase = self.refinement_phase(fd)
        out_phase_complex_ = self.output_phase(fr_phase)

        clean_mag = torch.tanh(out_mag_mask) * spectrogram_mag

        clean_complex = torch.complex(clean_mag * torch.cos(spectrogram_phase) + out_phase_complex_[:, 0, :, :],
                                      clean_mag * torch.sin(spectrogram_phase) + out_phase_complex_[:, 1, :, :]).squeeze(1)

        return clean_complex


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with torch.autograd.set_detect_anomaly(True):
        model = ComplexRestormer().to(device=device)
        random_audio = torch.randn((1, model.hop_length * 159), requires_grad=True).to(device=device)
        random_audio.retain_grad()

        print("input shape: {}".format(random_audio.shape))

        only_stft_output_complex = model.only_stft_forward(random_audio)
        print("only stft output shape: {}".format(only_stft_output_complex.shape))

        only_istft_output = model.only_istft_forward(only_stft_output_complex)
        print("only istft output shape: {}".format(only_istft_output.shape))

        print("stft-istft error: {}".format((random_audio - only_istft_output).abs().sum()))

        macs, params = ptflops.get_model_complexity_info(model, (model.hop_length * 159,),
                                                         as_strings=True,
                                                         print_per_layer_stat=False,
                                                         verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        model.train()
        model_output = model(random_audio)[2]
        print("model output shape: {}".format(random_audio.shape))

        model_output.mean().backward()

        print(random_audio.grad)

        random_spec = torch.randn((1, 160, 160), dtype=torch.complex64).to(device=device)
        random_spec_istft = model.only_istft_forward(random_spec)
        random_spec_istft_stft = model.only_stft_forward(random_spec_istft)

        print("istft-stft error: {}".format((random_spec_istft_stft - random_spec).abs().sum()))
