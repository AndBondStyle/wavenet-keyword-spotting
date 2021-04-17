from typing import NamedTuple, List, Tuple
import torch.nn as nn
import torch


class WaveNetKWS(nn.Module):
    """
    WaveNet-based keyword spotting network

    References:
    - https://arxiv.org/pdf/1609.03499.pdf
    - https://arxiv.org/pdf/1811.07684.pdf
    - https://github.com/ibab/tensorflow-wavenet
    - https://github.com/vincentherrmann/pytorch-wavenet
    - https://github.com/Dankrushen/Wavenet-PyTorch

    Constructor args:
    - num_groups, group_size - number of groups and number of blocks in each group.
    Total number of blocks: num_groups * group_size. Each block dilation factor is
    1, 2, 4, ... (2 ^ (group_size - 1)), 1, 2, 4, ... (repeated <num_groups> times).
    - dilations - list of custom dilation factors (instead of default sequence described
    above). If specified, num_groups and group_size arguments are ignored.
    - kernel_size - kernel size of dilated convolutions
    - input_channels - size of input features dimension (axis=1)
    - residual_channels - size of residual features dimension (axis=1)
    - dilation_channels - size of dilation features dimension (axis=1)
    - skip_channels - size of skip features dimension (axis=1)
    - out_hidden_channels - size of output block hidden features dimension (axis=1)
    - out_hidden_size - "wavenet pyramid" output size over time dimension (axis=2)
    By default, equals to last block dilation factor, and cannot be smaller.
    - out_classes - number of output classes

    Dynamic attributes:
    - input_size - model's receptive field (input size over time dimension), calculated
    based on out_hidden_size, dilation factors and kernel_size. See __init__ for
    details.
    - blocks - nn.ModuleList of wavenet blocks. Each block has .meta attribute,
    containing BlockMeta object (see below). Blocks are ordered bottom-to-top, first
    block being closest to model inputs.
    """

    class BlockMeta(NamedTuple):
        """
        WaveNet block metadata wrapper
        - dilation - dilation factor for 1D-convolutions
        - input_size - input size over time dimension (axis=2)
        - output_size - output size over time dimension (axis=2)
        """

        dilation: int
        input_size: int
        output_size: int

    def __init__(
        self,
        num_groups: int = 10,
        group_size: int = 4,
        dilations: List[int] = None,
        kernel_size: int = 2,
        input_channels: int = 40,
        residual_channels: int = 16,
        dilation_channels: int = 16,
        skip_channels: int = 32,
        out_hidden_channels: int = 128,
        out_hidden_size: int = None,
        out_classes: int = 2,
    ):
        super().__init__()
        self.dilations = dilations or [2 ** i for i in range(group_size)] * num_groups
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.out_hidden_channels = out_hidden_channels
        self.out_hidden_size = out_hidden_size or self.dilations[-1]
        self.out_classes = out_classes

        # Structural params calculation (traversing from top to bottom)
        inputs = self.out_hidden_size
        block_meta = []
        for dilation in reversed(self.dilations):
            outputs = inputs
            inputs += dilation * (self.kernel_size - 1)
            block_meta.append(self.BlockMeta(dilation, inputs, outputs))
        block_meta.reverse()
        self.input_size = block_meta[0].input_size

        # Input layer initialization (input_channels -> residual_channels)
        self.input_conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=1,
        )

        # WaveNet blocks initialization
        self.blocks = nn.ModuleList()
        for meta in block_meta:
            block = nn.ModuleDict(
                dict(
                    filter_conv=nn.Conv1d(
                        in_channels=self.residual_channels,
                        out_channels=self.dilation_channels,
                        kernel_size=self.kernel_size,
                        dilation=meta.dilation,
                    ),
                    gate_conv=nn.Conv1d(
                        in_channels=self.residual_channels,
                        out_channels=self.dilation_channels,
                        kernel_size=self.kernel_size,
                        dilation=meta.dilation,
                    ),
                    skip_conv=nn.Conv1d(
                        in_channels=self.dilation_channels,
                        out_channels=self.skip_channels,
                        kernel_size=1,
                    ),
                    residual_conv=nn.Conv1d(
                        in_channels=self.dilation_channels,
                        out_channels=self.residual_channels,
                        kernel_size=1,
                    ),
                    batch_norm=nn.BatchNorm1d(
                        num_features=self.residual_channels,
                    ),
                )
            )
            block.meta = meta
            self.blocks.append(block)

        # Output layers initialization
        self.output_convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.skip_channels,
                out_channels=self.out_hidden_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.out_hidden_channels,
                out_channels=self.out_classes,
                kernel_size=1,
            ),
        )
        self.output_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.out_classes * self.input_size,
                out_features=self.out_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-window inference with batch support
        Input shape: (<batch size>, model.input_channels, model.input_size)
        Output shape: (<batch size>, model.out_classes)
        """
        assert x.shape[1:] == (self.input_channels, self.input_size)
        x = self.input_conv(x)
        skip_shape = (x.size(0), self.skip_channels, x.size(2))
        skip = torch.zeros(skip_shape, dtype=x.dtype, device=x.device)

        for block in self.blocks:
            assert x.size(2) == block.meta.input_size
            residual = x
            filter_out = torch.tanh(block["filter_conv"](x))
            gate_out = torch.sigmoid(block["gate_conv"](x))
            x = filter_out * gate_out
            x_skip = block["skip_conv"](x)
            skip[:, :, -x_skip.size(2) :] += x_skip
            x = block["residual_conv"](x)
            x = residual[:, :, -x.size(2) :] + x
            x = block["batch_norm"](x)
            assert x.size(2) == block.meta.output_size

        out = self.output_convs(skip)
        out = self.output_dense(out)
        return out

    def lazy_conv(self, x, conv, buffer, dilation):
        """
        Lazy convolution call, computing only one sample (axis=1 column), based on one
        new sample and history buffer. Returns new sample and updated buffer.
        :param x: single input sample, shape = (1, <conv in channels>, 1)
        :param conv: 1D convolution layer / function
        :param buffer: input history buffer, at least (dilation * (kernel_size - 1) + 1)
        samples in length (this length is required to apply convolution).
        :param dilation: convolution dilation factor
        :return: (new convolution output, new history buffer)
        """
        assert x.size(2) == 1
        buffer = torch.cat((buffer[:, :, 1:], x), dim=2)
        window_size = dilation * (self.kernel_size - 1) + 1
        out = conv(buffer[:, :, -window_size:])
        assert out.size(2) == 1
        return out, buffer

    def init_buffers(self):
        """
        Initializes history buffers for stream inference (see below)
        :return: dict of buffers (just pass them into stream_forward)
        """
        shapes = dict(out=(1, self.out_classes, self.input_size))
        for i, block in enumerate(self.blocks):
            for sub in ("gate", "filter"):
                shape = (1, self.residual_channels, block.meta.input_size)
                shapes[f"block-{i}-{sub}"] = shape
        # TODO: Looks a bit funky, could be broken by quantization
        device = self.input_conv.weight.device
        dtype = self.input_conv.weight.dtype
        buffers = {
            key: torch.zeros(shape, dtype=dtype, device=device)
            for key, shape in shapes.items()
        }
        return buffers

    def stream_forward(
        self, x: torch.Tensor, buffers: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Stream inference using cached dilated buffers
        :param x: one input feature tensor, shape: (1, model.input_channels, 1)
        :param buffers: buffers dict, can be generated with model.init_buffers method
        :return: (model output: shape: (1, model.out_classes), new buffers dict)
        """
        assert x.size(0) == x.size(2) == 1
        # Just copying dict items, no tensors involved
        buffers = buffers.copy()
        skip_shape = (1, self.skip_channels, 1)
        skip = torch.zeros(skip_shape, dtype=x.dtype, device=x.device)
        x = self.input_conv(x)

        for i, block in enumerate(self.blocks):
            assert x.size(2) == 1
            residual = x

            dilated_outs = {}
            for sub in ("gate", "filter"):
                key = f"block-{i}-{sub}"
                dilated_outs[sub], buffers[key] = self.lazy_conv(
                    x, block[f"{sub}_conv"], buffers[key], block.meta.dilation
                )

            filter_out = torch.tanh(dilated_outs["filter"])
            gate_out = torch.sigmoid(dilated_outs["gate"])
            x = filter_out * gate_out
            x_skip = block["skip_conv"](x)
            skip += x_skip
            x = block["residual_conv"](x)
            x = residual + x
            x = block["batch_norm"](x)
            assert x.size(2) == 1

        out = self.output_convs(skip)
        buffers["out"] = torch.cat((buffers["out"][:, :, 1:], out), dim=2)
        buffers["out"][:, :, 0] = 0
        assert buffers["out"].size(2) == self.input_size
        out = self.output_dense(buffers["out"])
        return out, buffers
