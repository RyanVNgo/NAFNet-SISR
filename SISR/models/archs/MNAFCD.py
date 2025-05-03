
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

class MNAFCDModel(nn.Module):
    def __init__(self, iterations, device='cpu'):
        super().__init__()
        self.net = MNAFD(3, 16, [1, 1, 1]).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=iterations,
            eta_min=1e-7
        )
        self.scaler = GradScaler()

    def forward(self, x):
        with autocast(self.device):
            return self.net(x)

    def update(self, pred, target):
        with autocast(self.device):
            T_pred = self.net(target.to(self.device))
            P_pred = self.net(pred.detach().to(self.device))
        d_pred_diff = T_pred - P_pred
        d_loss = nn.BCEWithLogitsLoss()(d_pred_diff, torch.ones_like(d_pred_diff))
        self.optimizer.zero_grad()
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return d_loss 

    def adversarial_loss(self, pred, target):
        with autocast(self.device):
            T_pred = self.net(target.to(self.device))
            P_pred = self.net(pred.detach().to(self.device))
        adv_pred_diff = P_pred - T_pred
        adv_loss = nn.BCEWithLogitsLoss()(adv_pred_diff, torch.ones_like(adv_pred_diff))
        return adv_loss


class MNAFD(nn.Module):
    def __init__(self, c_in=3, width=16, down_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels = c_in,
            out_channels = width,
            kernel_size = 3,
            padding = 1
        )

        curr_channels = width

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for num in down_blk_nums:
            self.blocks.append(
                nn.Sequential(
                    *[NAFNetBlock(c_in = curr_channels) for _ in range(num)]
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


        self.ending = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(curr_channels, 1, 1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.intro(input)
        for encoder, down in zip(self.blocks, self.downs):
            input = encoder(input)
            input = down(input)
        input = self.ending(input)
        return input


class NAFNetBlock(nn.Module):
    def __init__(self, c_in, dw_expand=2, ffn_expand=2, dropout=0.0):
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
            padding = 1,
            groups = dw_channels
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

'''
model = MNAFCD(3, 16, [1, 1, 1])
input = torch.randn(1, 3, 256, 256)
output = model(input)
print(output.shape)

loss_fn = nn.BCEWithLogitsLoss()

loss = loss_fn(output, torch.ones_like(output))
print(loss)
'''
