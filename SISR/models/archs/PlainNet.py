
import torch
import torch.nn as nn


class PlainNet(nn.Module):
    def __init__(self, c_in=3, width=16, mid_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels = c_in,
            out_channels = width,
            kernel_size = 3,
            padding = 1
        )

        curr_channels = width

        # Construct the encoding path
        self.encoding_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for num in enc_blk_nums:
            self.encoding_blocks.append(
                nn.Sequential(
                    *[PlainNetBlock(c_in = curr_channels) for _ in range(num)]
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
            nn.Sequential(
                *[PlainNetBlock(c_in = curr_channels) for _ in range(mid_blk_num)]
            )
        )

        # Construct the decoding path
        self.decoding_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in dec_blk_nums:
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
                nn.Sequential(
                    *[PlainNetBlock(c_in = curr_channels) for _ in range(num)]
                )
            )

        self.ending = nn.Conv2d(
            in_channels = width,
            out_channels = c_in,
            kernel_size = 3,
            padding = 1
        )

        # Appended Upscaling Block
        self.upscale_block = nn.Sequential(
            nn.Conv2d(c_in, c_in * 4, kernel_size = 3, padding = 1),
            nn.PixelShuffle(2)
        )

    def forward(self, input):
        enc_outs = []

        input = self.intro(input)

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

        input = self.ending(input)

        input = self.upscale_block(input)
        return input 


class PlainNetBlock(nn.Module):
    def __init__(self, c_in, dw_expand=1, ffn_expand=2, dropout=0.0):
        super().__init__()

        # First stage of block
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
            in_channels = dw_channels, 
            out_channels = c_in,
            kernel_size = 1,
            padding = 0
        )

        # First dropout
        self.dropout_1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Second stage of block
        ffn_channels = c_in * ffn_expand 
        self.conv_4 = nn.Conv2d(
            in_channels = c_in, 
            out_channels = ffn_channels,
            kernel_size = 1,
            padding = 0
        )
        self.conv_5 = nn.Conv2d(
            in_channels = ffn_channels, 
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
        x = self.conv_1(input)
        x = self.conv_2(x)
        x = nn.ReLU()(x)
        x = self.conv_3(x)

        x = self.dropout_1(x)

        y = input + x * self.beta

        x = self.conv_4(y)
        x = nn.ReLU()(x)
        x = self.conv_5(x)

        x = self.dropout_2(x)

        return y + x * self.gamma


