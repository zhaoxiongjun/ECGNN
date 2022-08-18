from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn.gnn_models import Model


class attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim , dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, enc_output):
        
      #  enc_output = enc_output.transpose(2, 1)
        
        energy = torch.tanh(self.attn(enc_output))
        
        attention = self.v(energy)
        
        scores = F.softmax(attention, dim=1)
        out = enc_output*scores

        return torch.sum(out, dim=1), scores


class ResBlock(nn.Module):
    """A class used to build a Residual block.

    Attributes
    ----------
    in_channel: int
        The number of input channels.
    out_channel: int
        The number of output channels.
    kernel_size: int
        The size of the kernel for 1D-convolution.
    stride: int
        The stride for 1D-convolution.
    padding: int
        The padding for 1D-convolution.
    downsample: bool, optional
        If True, downsamples the input. (default: None)

    Methods
    -------
    forward(x)
        Calculates the output of the Residual block.

    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(7 if not downsample else 8),
            stride=(1 if not downsample else 2),
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=False)  # 是否需要2个relu, inplace mean?
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )

        self.conv_down = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
      #  out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample :
            identity = self.conv_down(x)
        out += identity

        out = self.relu(out)
        out = self.bn2(out)

        return out


class ECGNN(nn.Module):
    def __init__(self, args, in_channels: int = 12, num_classes: int = 5):
        super().__init__()

        self.beat_len = 50
        self.start_filters = 32
        self.num_classes = args.num_classes

        self.conv1 = nn.Conv1d(
            in_channels=1,  # 1
            out_channels=32,
            kernel_size=7,  ### pytorch 偶数时如何保持前后dim一致
            stride=1,
            padding=3,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=False)
        self.block = self._make_layer()

        self.beat_attn = attention(enc_hid_dim=128, dec_hid_dim=64)

        self.biLSTM = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.rhythm_attn = attention(enc_hid_dim=128, dec_hid_dim=64)

        self.gcn = Model(args)

        self.channel_attn = attention(enc_hid_dim=128, dec_hid_dim=64)
        self.fc = nn.Linear(in_features=160, out_features=self.num_classes)  # need change


    def _make_layer(self, blocks_list: list = [2, 2, 2]) -> List[ResBlock]:
        layers = []
        downsample = None

        num_filters = 32
        old_filters = num_filters
        for i in range(len(blocks_list)):
            num_blocks = blocks_list[i]
            for j in range(num_blocks):
                downsample = True if j == 0 and i != 0 else False
                layers.append(
                    ResBlock(
                        in_channels=old_filters,
                        out_channels=num_filters,
                        downsample=downsample,
                    )
                )
                old_filters = num_filters
            num_filters *= 2
            

        return nn.Sequential(*layers)


    def forward(self, data):
        x, edge_index, batch = data.x.double(), data.edge_index, data.batch

        x = x.reshape(-1, self.beat_len).unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.block(x)
        x = x.transpose(2, 1)
        x, _ = self.beat_attn(x)

        x = x.reshape(-1, int(1000/50), 128)
        x, _ = self.biLSTM(x)
        x, _ = self.rhythm_attn(x)

        x = x.reshape(-1, 12, 128)

        gcn_out = self.gcn(x.reshape(-1, 128, 1).squeeze(2), edge_index, batch)  #(b, 32)

        x, _ = self.channel_attn(x)
        # with gcn
        new_x = torch.cat([x, gcn_out], dim=1)
        # without gcn
      #  new_x = x
        # only gcn
        # new_x = gcn_out
        out = self.fc(new_x)

        return out


       





    