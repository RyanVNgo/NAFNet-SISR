
import torch
import torch.nn as nn


class SRNAFNet(nn.Module):
    def __init__(self, c_in=3, width=16, sfe_k_nums=[3,5,7], dfe_count=1, dfe_k=3, ufe_count=1, ufe_k=3, intro_k=3, ending_k=3, block_opts=None):
        super().__init__()
        dw_expand = 2
        ffn_expand = 2

        if block_opts is not None:
            dw_expand = block_opts.get('dw_expand', dw_expand)
            ffn_expand = block_opts.get('ffn_expand', ffn_expand)

        curr_channels = width

        # ---- Intro (Initial Channel Expansion) ----
        self.intro = nn.Conv2d(
            in_channels = c_in,
            out_channels = curr_channels,
            kernel_size = intro_k,
            padding = (intro_k - 1) // 2
        )

        # ---- Shallow Feature Extractors ----
        self.sfe_paths = nn.ModuleList()
        for k in sfe_k_nums:
            self.sfe_paths.append(
                NAFNetBlock(curr_channels, dw_expand, ffn_expand, k)
            )

        # ---- Deep Feature Extraction ----
        curr_channels = curr_channels * len(sfe_k_nums)
        self.dfe_sequence = nn.Sequential(
            *[NAFNetBlock(curr_channels, dw_expand, ffn_expand, dfe_k) for _ in range(dfe_count)]
        )

        # --- Upscaling (2x) ----
        self.upscale = nn.PixelShuffle(2)
        curr_channels = curr_channels // 4

        # ---- Upscaled Feature Extraction ----
        self.ufe_sequence = nn.Sequential(
            *[NAFNetBlock(curr_channels, dw_expand, ffn_expand, ufe_k) for _ in range(ufe_count)]
        )

        # ---- Ending ----
        self.ending = nn.Conv2d(
            in_channels = curr_channels,
            out_channels = c_in,
            kernel_size = ending_k,
            padding = (ending_k - 1) // 2
        )

        # Additional Stuff
        self.c_in = c_in
        self.padder_size = 1

    def forward(self, input):
        B, C, H, W = input.shape
        input = self.conform_image_size(input)

        # ---- Intro (Initial Channel Expansion) ----
        input = self.intro(input)

        # ---- Shallow Feature Extraction ----
        sfe_extracts = []
        for sfe in self.sfe_paths:
            sfe_extracts.append(sfe(input))
        sfe_concat = torch.concat(sfe_extracts, dim=1)

        # ---- Deep Feature Extraction ----
        input = self.dfe_sequence(sfe_concat)

        # ---- Upscaling (2x) ----
        input = input + sfe_concat
        input = self.upscale(input)

        # ---- Upscaled Feature Extraction ----
        input = self.ufe_sequence(input)

        # ---- Ending ----
        input = self.ending(input)

        return input

    def conform_image_size(self, img):
        _, _, h, w = img.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        img = nn.functional.pad(img, (0, mod_pad_w, 0, mod_pad_h))
        return img


class NAFNetBlock(nn.Module):
    def __init__(self, c_in, dw_expand=2, ffn_expand=2, kernel_size=1):
        super().__init__()

        # First stage of block
        self.norm_1 = LayerNorm2d(c_in)
        
        dw_channels = c_in * dw_expand
        self.conv_1 = nn.Conv2d(
            in_channels = c_in, 
            out_channels = dw_channels,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2
        )
        self.conv_2 = nn.Conv2d(
            in_channels = dw_channels, 
            out_channels = dw_channels,
            kernel_size = (kernel_size * 2) + 1,
            padding = kernel_size,
            groups = dw_channels
        )
        self.conv_3 = nn.Conv2d(
            in_channels = dw_channels // 2, 
            out_channels = c_in,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2
        )

        self.simple_gate = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels = dw_channels // 2, 
                out_channels = dw_channels // 2, 
                kernel_size=1, 
            ),
        )

        # Second stage of block
        self.norm_2 = LayerNorm2d(c_in)
        
        ffn_channels = c_in * ffn_expand 
        self.conv_4 = nn.Conv2d(
            in_channels = c_in, 
            out_channels = ffn_channels,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2
        )
        self.conv_5 = nn.Conv2d(
            in_channels = ffn_channels // 2, 
            out_channels = c_in,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2
        )

        scalar_init = 1e-2
        self.beta = nn.Parameter(torch.full((1, c_in, 1, 1), scalar_init), requires_grad=True)
        self.gamma = nn.Parameter(torch.full((1, c_in, 1, 1), scalar_init), requires_grad=True)

    def forward(self, input):
        x = self.norm_1(input)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.simple_gate(x)
        x = x * self.sca(x)
        x = self.conv_3(x)

        y = input + x * self.beta

        x = self.norm_2(y)
        x = self.conv_4(x)
        x = self.simple_gate(x)
        x = self.conv_5(x)

        return y + x * self.gamma


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def nafnet_weight_init(module, scale=0.1):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        module.weight.data *= scale
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, LayerNorm2d):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()


