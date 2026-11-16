"""EfficientSleepNet compact CNN–attention–BiLSTM architecture.

Input: [batch, 1, 3000] for one 30-second EEG epoch.
Output: [batch, 5] sleep-stage logits.
Grouped convolutions reduce the parameter count while preserving feature widths.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from config import (
    IN_CHANNELS, MRCNN_CHANNELS, MRCNN_SHORT_KERNELS,
    MRCNN_LONG_KERNELS, MRCNN_POOL_SIZE, MRCNN_DROPOUT,
    MRCNN_SHORT_GROUPS, MRCNN_LONG_GROUPS, MRCNN_LONG_CONV3_POINTWISE,
    COMPACT_CONVOLUTIONS,
    CARA_HEADS, CARA_PROJECTION_DIM, CARA_REDUCTION,
    CARA_GROUPS, CARA_DEPTHWISE_KERNEL, CARA_DROPOUT,
    RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, RNN_DROPOUT, ATTENTION_POOL_DIM,
    NUM_CLASSES, INPUT_LENGTH,
)


class SameConv1d(nn.Module):
    """Exact length-preserving stride-1 convolution, including even kernels."""
    def __init__(self, ci, co, kernel, groups=1, bias=False):
        super().__init__()
        self.kernel = int(kernel)
        self.conv = nn.Conv1d(ci, co, self.kernel, stride=1, groups=groups, bias=bias)

    def forward(self, x):
        p = self.kernel - 1
        return self.conv(F.pad(x, (p // 2, p - p // 2)))


class CausalDepthwiseConv1d(nn.Module):
    """Local temporal encoding uses only past/current values."""
    def __init__(self, channels, kernel):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)

    def forward(self, x):
        return self.conv(F.pad(x, (self.kernel - 1, 0)))


class ConvBNGELU(nn.Module):
    def __init__(self, ci, co, kernel, groups=1, pointwise_mixing=False):
        super().__init__()
        # Grouped Conv1D reduces parameters without changing output dimensions.
        main = SameConv1d(ci, co, kernel, groups=groups)
        # 1x1 mixing after long Conv3 re-connects the convolution groups.
        self.conv = (nn.Sequential(main, nn.Conv1d(co, co, 1, bias=False))
                     if pointwise_mixing else main)
        self.bn = nn.BatchNorm1d(co)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MRCNNBranch(nn.Module):
    """Branch: Conv1 -> Pool1 -> Conv2 -> Pool2 -> Conv3."""
    def __init__(self, kernels, dropout, groups=(1, 1, 1), conv3_pointwise=False):
        super().__init__()
        c1,c2,c3 = MRCNN_CHANNELS
        self.conv1 = ConvBNGELU(IN_CHANNELS, c1, kernels[0], groups[0])
        self.pool1 = nn.MaxPool1d(MRCNN_POOL_SIZE, stride=MRCNN_POOL_SIZE)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = ConvBNGELU(c1, c2, kernels[1], groups[1])
        self.pool2 = nn.MaxPool1d(MRCNN_POOL_SIZE, stride=MRCNN_POOL_SIZE)
        self.conv3 = ConvBNGELU(c2, c3, kernels[2], groups[2], conv3_pointwise)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.conv1(x)))
        x = self.pool2(self.conv2(x))
        return self.conv3(x)


class MRCNN(nn.Module):
    def __init__(self, compact=COMPACT_CONVOLUTIONS):
        super().__init__()
        self.short_branch = MRCNNBranch(
            MRCNN_SHORT_KERNELS, MRCNN_DROPOUT,
            groups=MRCNN_SHORT_GROUPS if compact else (1, 1, 1))
        self.long_branch = MRCNNBranch(
            MRCNN_LONG_KERNELS, MRCNN_DROPOUT,
            groups=MRCNN_LONG_GROUPS if compact else (1, 1, 1),
            conv3_pointwise=MRCNN_LONG_CONV3_POINTWISE if compact else False)
        self.dropout = nn.Dropout(MRCNN_DROPOUT)
        self.out_channels = MRCNN_CHANNELS[-1] * 2

    def forward(self, x):
        a = self.short_branch(x)
        b = self.long_branch(x)
        if a.shape[-1] != b.shape[-1]:
            raise RuntimeError(f"Unequal MRCNN branch lengths: {a.shape} and {b.shape}")
        return self.dropout(torch.cat([a, b], dim=1))


class MultiHeadTemporalAttention(nn.Module):
    """Five 51-dimensional heads, followed by projection to 256 channels."""
    def __init__(self, channels=256, heads=CARA_HEADS, inner=CARA_PROJECTION_DIM):
        super().__init__()
        if inner % heads != 0:
            raise ValueError("Internal attention dimension must be divisible by heads")
        self.heads = heads
        self.head_dim = inner // heads
        self.inner = inner
        self.q = nn.Linear(channels, inner, bias=False)
        self.k = nn.Linear(channels, inner, bias=False)
        self.v = nn.Linear(channels, inner, bias=False)
        self.out = nn.Linear(inner, channels, bias=False)
        self.dropout = nn.Dropout(CARA_DROPOUT)

    def forward(self, x):
        b, c, t = x.shape
        xt = x.transpose(1, 2)
        def split(z):
            return z.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        q,k,v = split(self.q(xt)),split(self.k(xt)),split(self.v(xt))
        # Numerically stable attention, avoids materializing [B,H,T,T] where supported.
        if hasattr(F, "scaled_dot_product_attention"):
            a = F.scaled_dot_product_attention(q, k, v,
                dropout_p=CARA_DROPOUT if self.training else 0.0)
        else:
            s = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
            a = torch.matmul(self.dropout(s.softmax(dim=-1)), v)
        a = a.transpose(1,2).reshape(b,t,self.inner)
        return self.out(a).transpose(1,2)


class CARA(nn.Module):
    """Depthwise refinement, positional encoding, attention, gating and residual."""
    def __init__(self, channels=256):
        super().__init__()
        if channels % CARA_GROUPS:
            raise ValueError("CARA_GROUPS must divide feature dimension")
        hidden = max(1, channels // CARA_REDUCTION)
        self.refine = nn.Sequential(
            SameConv1d(channels, channels, CARA_DEPTHWISE_KERNEL, groups=channels),
            nn.SiLU(), nn.GroupNorm(CARA_GROUPS,channels),
        )
        self.position = CausalDepthwiseConv1d(channels,CARA_DEPTHWISE_KERNEL)
        self.attn = MultiHeadTemporalAttention(channels)
        self.gate = nn.Sequential(
            nn.Linear(2*channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        local = self.refine(x)
        temporal = self.attn(self.position(local))
        s_avg = temporal.mean(dim=-1)
        s_max = temporal.amax(dim=-1)
        gates = self.gate(torch.cat((s_avg,s_max),dim=-1)).unsqueeze(-1)
        return x + temporal * gates


class TemporalAttentionPool(nn.Module):
    """Learned temporal attention pooling for a single five-class prediction."""
    def __init__(self, input_dim=256, attention_dim=ATTENTION_POOL_DIM):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(input_dim,attention_dim), nn.Tanh(),
                                   nn.Linear(attention_dim,1))

    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1),dim=1)
        return torch.sum(x * w.unsqueeze(-1),dim=1), w


class EfficientSleepNet(nn.Module):
    def __init__(self, input_length=INPUT_LENGTH, num_classes=NUM_CLASSES,
                 compact=COMPACT_CONVOLUTIONS):
        super().__init__()
        self.input_length = input_length
        self.mrcnn = MRCNN(compact=compact)
        self.cara = CARA(self.mrcnn.out_channels)
        self.birnn = nn.LSTM(
            input_size=self.mrcnn.out_channels, hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_NUM_LAYERS, bidirectional=True, batch_first=True,
            dropout=RNN_DROPOUT if RNN_NUM_LAYERS > 1 else 0.0,
        )
        self.temporal_pool = TemporalAttentionPool(2*RNN_HIDDEN_SIZE)
        self.classifier = nn.Linear(2*RNN_HIDDEN_SIZE,num_classes)

    def forward(self, x, return_attention=False):
        if x.ndim != 3 or x.shape[1] != IN_CHANNELS or x.shape[-1] % INPUT_LENGTH != 0:
            raise ValueError(f"Expected [B,1,3000*L] input; got {tuple(x.shape)}")
        feat = self.cara(self.mrcnn(x))
        hidden,_ = self.birnn(feat.transpose(1,2))
        context,weights = self.temporal_pool(hidden)
        logits = self.classifier(context)
        return (logits,weights) if return_attention else logits


def efficientsleepnet(input_length=INPUT_LENGTH,num_classes=NUM_CLASSES,
                     compact=COMPACT_CONVOLUTIONS):
    return EfficientSleepNet(input_length,num_classes,compact=compact)


if __name__ == "__main__":
    from utils import count_parameters
    with torch.no_grad():
        model = EfficientSleepNet().eval()
        x = torch.randn(1,1,INPUT_LENGTH)
        feat = model.mrcnn(x)
        print('MRCNN:',tuple(feat.shape),'logits:',tuple(model(x).shape))
        print('Params:',count_parameters(model))
