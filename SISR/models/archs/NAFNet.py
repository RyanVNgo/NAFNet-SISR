
import torch
import torch.nn as nn


class NAFNet(nn.Module):
    def __init__(self, c_in, depth = 3, dw_expand = 2, ffn_expand = 2, dropout = 0.0):
        super().__init__()
        curr_channels = c_in

        # Construct the encoding path
        self.encoding_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for _ in range(depth - 1):
            self.encoding_blocks.append(
                NAFNetBlock(
                    c_in = curr_channels,
                    dw_expand = dw_expand,
                    ffn_expand = ffn_expand,
                    dropout = dropout
                )
            )
            self.downs.append(
                nn.Conv2d(
                    in_channels = curr_channels,
                    out_channels = curr_channels * 2,
                    kernel_size = 2,
                    stride = 2
                )
            )
            curr_channels = curr_channels * 2

        # Construct the middle path
        self.middle_blocks = nn.ModuleList()
        self.middle_blocks.append(
            NAFNetBlock(
                c_in = curr_channels,
                dw_expand = dw_expand,
                ffn_expand = ffn_expand,
                dropout = dropout
            )
        )

        # Construct the decoding path
        self.decoding_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for _ in range(depth - 1):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels = curr_channels, 
                        out_channels = curr_channels * 2, 
                        kernel_size = 1,
                    ),
                    nn.PixelShuffle(2)
                )
            )
            curr_channels = curr_channels // 2
            self.decoding_blocks.append(
                NAFNetBlock(
                    c_in = curr_channels,
                    dw_expand = dw_expand,
                    ffn_expand = ffn_expand,
                    dropout = dropout
                )
            )

        # Appended Upscaling Block
        self.upscale_block = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels * 4, kernel_size = 3, padding = 1),
            nn.PixelShuffle(2)
        )


    def forward(self, input):
        enc_outs = []

        for encoder, down in zip(self.encoding_blocks, self.downs):
            input = encoder(input)
            enc_outs.append(input)
            input = down(input)

        for middle_block in self.middle_blocks:
            input = middle_block(input)

        for decoder, up, enc_skip in zip(self.decoding_blocks, self.ups, enc_outs[::-1]):
            input = up(input)
            input = input + enc_skip
            input = decoder(input)

        input = self.upscale_block(input)
        return input 


class NAFNetBlock(nn.Module):
    def __init__(self, c_in, dw_expand = 2, ffn_expand = 2, dropout = 0.0):
        super().__init__()

        # First stage of block
        self.norm_1 = LayerNorm2d(c_in)
        
        dw_channels = c_in * dw_expand
        self.conv_1 = nn.Conv2d(
            in_channels = c_in, 
            out_channels = dw_channels,
            kernel_size = 1,
            padding = 0
        )
        self.conv_2 = nn.Conv2d(
            in_channels = dw_channels, 
            out_channels = dw_channels,
            kernel_size = 3,
            padding = 1
        )
        self.conv_3 = nn.Conv2d(
            in_channels = dw_channels // 2, 
            out_channels = c_in,
            kernel_size = 1,
            padding = 0
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

        # First dropout
        self.dropout_1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Second stage of block
        self.norm_2 = LayerNorm2d(c_in)
        
        ffn_channels = c_in * ffn_expand 
        self.conv_4 = nn.Conv2d(
            in_channels = c_in, 
            out_channels = ffn_channels,
            kernel_size = 1,
            padding = 0
        )
        self.conv_5 = nn.Conv2d(
            in_channels = ffn_channels // 2, 
            out_channels = c_in,
            kernel_size = 1,
            padding = 0
        )

        # First dropout
        self.dropout_2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Not really sure wtf these are, they were in the open archs
        self.beta = nn.Parameter(torch.zeros((1, c_in, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c_in, 1, 1)), requires_grad=True)

    def forward(self, input):
        x = self.norm_1(input)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.simple_gate(x)
        x = x * self.sca(x)
        x = self.conv_3(x)

        x = self.dropout_1(x)

        y = input + x * self.beta

        x = self.norm_2(y)
        x = self.conv_4(x)
        x = self.simple_gate(x)
        x = self.conv_5(x)

        x = self.dropout_2(x)

        return y + x * self.gamma


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


