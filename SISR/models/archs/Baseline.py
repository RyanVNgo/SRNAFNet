
import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, c_in=3, width=16, mid_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], intro_k=3, ending_k=3, block_opts=None):
        super().__init__()
        dw_expand = 1
        ffn_expand = 2
        block_k = 1

        if block_opts is not None:
            dw_expand = block_opts.get('dw_expand', dw_expand)
            ffn_expand = block_opts.get('ffn_expand', ffn_expand)
            block_k = block_opts.get('kernel_size', block_k)

        self.intro = nn.Conv2d(
            in_channels = c_in,
            out_channels = width,
            kernel_size = intro_k,
            padding = (intro_k - 1) // 2
        )

        curr_channels = width

        # Construct the encoding path
        self.encoding_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for num in enc_blk_nums:
            self.encoding_blocks.append(
                nn.Sequential(
                    *[BaselineBlock(curr_channels, dw_expand, ffn_expand, block_k) for _ in range(num)]
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
                *[BaselineBlock(curr_channels, dw_expand, ffn_expand, block_k) for _ in range(mid_blk_num)]
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
                        bias = False
                    ),
                    nn.PixelShuffle(2)
                )
            )
            curr_channels = curr_channels // 2
            self.decoding_blocks.append(
                nn.Sequential(
                    *[BaselineBlock(curr_channels, dw_expand, ffn_expand, block_k) for _ in range(num)]
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

        # Modified Ending
        self.final_reconstruction = nn.Sequential(
            nn.Conv2d(
                in_channels = width, 
                out_channels = c_in * 4, 
                kernel_size = ending_k,
                padding = (ending_k - 1) // 2
            ),
            nn.PixelShuffle(2)
        )

        # Additional Stuff
        self.c_in = c_in
        self.padder_size = 2 ** len(self.encoding_blocks)

    def forward(self, input):
        B, C, H, W = input.shape
        input = self.conform_image_size(input)

        input = self.intro(input)

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

        # input = self.ending(input)
        # input = self.upscale_block(input)

        input = self.final_reconstruction(input)

        return input

    def conform_image_size(self, img):
        _, _, h, w = img.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        img = nn.functional.pad(img, (0, mod_pad_w, 0, mod_pad_h))
        return img


class BaselineBlock(nn.Module):
    def __init__(self, c_in, dw_expand=1, ffn_expand=2, kernel_size=1, dropout=0.0):
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
            in_channels = dw_channels, 
            out_channels = c_in,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels = dw_channels, 
                out_channels = dw_channels // 2, 
                kernel_size=1, 
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels = dw_channels // 2, 
                out_channels = dw_channels,
                kernel_size=1,
            ),
            nn.Sigmoid()
        )

        # First dropout
        self.dropout_1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

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
            in_channels = ffn_channels, 
            out_channels = c_in,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2
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
        x = nn.GELU()(x)
        x = x * self.channel_attention(x)
        x = self.conv_3(x)

        x = self.dropout_1(x)

        y = input + x * self.beta

        x = self.norm_2(y)
        x = self.conv_4(x)
        x = nn.GELU()(x)
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


