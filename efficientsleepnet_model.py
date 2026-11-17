"""Compact CNN–CARA–BiLSTM encoder with attentive LSTM decoder.

Input: [batch, 1, 3000] for one 30-second EEG epoch.
Default output: [batch, 5] sleep-stage logits.
Optional autoregressive decoding supports multiple output steps.
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
    RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, RNN_DROPOUT,
    NUM_CLASSES, INPUT_LENGTH,
    DECODER_HIDDEN_SIZE, DECODER_NUM_LAYERS, DECODER_EMBED_DIM, DECODER_ATTENTION_DIM,
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


class AttentiveLSTMDecoder(nn.Module):
    """Two-layer autoregressive LSTM with encoder–decoder additive attention.

    Decoder state is initialized from both directions of the two-layer BiLSTM.
    The decoder begins with a learned START token; subsequent steps can use
    teacher-forced stage labels in training or the previous predicted stage.
    For a single 30-second EEG epoch the decoder runs exactly one step.
    """
    def __init__(self, encoder_dim, encoder_hidden, num_classes=NUM_CLASSES,
                 hidden_size=DECODER_HIDDEN_SIZE, num_layers=DECODER_NUM_LAYERS,
                 embed_dim=DECODER_EMBED_DIM, attn_dim=DECODER_ATTENTION_DIM):
        super().__init__()
        if num_layers != 2:
            raise ValueError('This architecture expects a two-layer decoder')
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.start_token = num_classes
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)
        self.encoder_key = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.decoder_query = nn.Linear(hidden_size, attn_dim)
        self.energy = nn.Linear(attn_dim, 1, bias=False)
        # Share compact state bridges across the two decoder layers.
        self.init_hidden = nn.Linear(2*encoder_hidden, hidden_size)
        self.init_cell = nn.Linear(2*encoder_hidden, hidden_size)
        self.cells = nn.ModuleList([
            nn.LSTMCell(embed_dim+encoder_dim, hidden_size),
            nn.LSTMCell(hidden_size, hidden_size),
        ])
        self.output = nn.Linear(hidden_size+encoder_dim, num_classes)

    def forward(self, encoder_outputs, encoder_state, steps=1, targets=None,
                teacher_forcing_ratio=0.0):
        """Return [B,steps,classes] logits and [B,steps,T] attention weights.

        targets may be [B,steps] stage indices; for step t>0, targets[:,t-1]
        may replace the prior predicted token with probability teacher_forcing_ratio.
        The teacher-forced path is disabled in evaluation mode.
        """
        if steps < 1:
            raise ValueError('steps must be at least 1')
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError('teacher_forcing_ratio must be between 0 and 1')
        b, _, _ = encoder_outputs.shape
        if targets is not None:
            if targets.shape != (b,steps):
                raise ValueError(f'targets must be [{b},{steps}], got {tuple(targets.shape)}')
            if (targets < 0).any() or (targets >= self.num_classes).any():
                raise ValueError('targets must contain valid sleep-stage indices')
        enc_h,enc_c=encoder_state
        if enc_h.shape[0] != self.num_layers*2 or enc_c.shape != enc_h.shape:
            raise ValueError('Expected bidirectional encoder hidden and cell states for two layers')
        # PyTorch BiLSTM state: layer0_f, layer0_b, layer1_f, layer1_b.
        hs=[]; cs=[]
        for layer in range(self.num_layers):
            both_h=torch.cat((enc_h[2*layer],enc_h[2*layer+1]),dim=-1)
            both_c=torch.cat((enc_c[2*layer],enc_c[2*layer+1]),dim=-1)
            hs.append(torch.tanh(self.init_hidden(both_h)))
            cs.append(torch.tanh(self.init_cell(both_c)))
        keys=self.encoder_key(encoder_outputs)
        prior=torch.full((b,),self.start_token,device=encoder_outputs.device,dtype=torch.long)
        logits=[]; attention=[]
        for step in range(steps):
            scores=self.energy(torch.tanh(keys + self.decoder_query(hs[-1]).unsqueeze(1))).squeeze(-1)
            weights=torch.softmax(scores,dim=1)
            context=torch.bmm(weights.unsqueeze(1),encoder_outputs).squeeze(1)
            token=self.embedding(prior)
            hs[0],cs[0]=self.cells[0](torch.cat([token,context],dim=-1),(hs[0],cs[0]))
            hs[1],cs[1]=self.cells[1](hs[0],(hs[1],cs[1]))
            step_logits=self.output(torch.cat((hs[-1],context),dim=-1))
            logits.append(step_logits)
            attention.append(weights)
            prior=step_logits.argmax(dim=-1)
            if (step < steps-1 and self.training and targets is not None
                    and teacher_forcing_ratio > 0):
                use_teacher=torch.rand(b,device=prior.device) < teacher_forcing_ratio
                prior=torch.where(use_teacher,targets[:,step],prior)
        return torch.stack(logits,dim=1),torch.stack(attention,dim=1)


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
        self.decoder = AttentiveLSTMDecoder(
            encoder_dim=2*RNN_HIDDEN_SIZE, encoder_hidden=RNN_HIDDEN_SIZE,
            num_classes=num_classes)

    def forward(self, x, return_attention=False, targets=None,
                teacher_forcing_ratio=0.0, decode_steps=1):
        if x.ndim != 3 or x.shape[1] != IN_CHANNELS or x.shape[-1] % INPUT_LENGTH != 0:
            raise ValueError(f'Expected [B,1,3000*L] input; got {tuple(x.shape)}')
        feat = self.cara(self.mrcnn(x))
        hidden,state = self.birnn(feat.transpose(1,2))
        logits,weights=self.decoder(hidden,state,steps=decode_steps,
                     targets=targets,teacher_forcing_ratio=teacher_forcing_ratio)
        # Existing training/evaluation API uses [B,5] for L=1.
        if decode_steps==1:
            logits,weights=logits[:,0],weights[:,0]
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
