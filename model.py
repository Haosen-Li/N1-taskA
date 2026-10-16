"""Small decoder-only Transformer; all mathematical layers are written out.

PyTorch supplies tensors, parameter registration and automatic differentiation.
No nn.Linear/Embedding/LayerNorm/Transformer/MultiheadAttention or fused attention.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class Config:
    vocab_size: int = 20
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    expansion: int = 4
    max_seq_len: int = 256


class Linear(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None

    def forward(self, x):
        y = x @ self.weight
        return y if self.bias is None else y + self.bias


class LayerNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x):
        # Normalize features of EACH token; never mix time or batch dimensions.
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(variance + self.eps) * self.scale + self.bias


def softmax(x, dim=-1):
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def gelu(x):
    # Standard tanh approximation to x * Phi(x).
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        # Fixed ALiBi-like distance prior (not learned, not in state_dict):
        # the first half of the heads have slope 0 (distance-neutral, free to
        # reach far back); the rest start out favoring nearby tokens.
        # For 4 heads: [0, 0, 0.5, 1.0].
        slopes = torch.zeros(cfg.n_heads)
        slopes[cfg.n_heads // 2:] = torch.linspace(0.5, 1.0, cfg.n_heads - cfg.n_heads // 2)
        self.register_buffer("distance_slopes", slopes, persistent=False)
        self.qkv = Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, window=None):
        b, t, c = x.shape
        # [B,T,3C] -> three [B,H,T,D] tensors; H*D = C.
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
                   for z in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        pos = torch.arange(t, device=x.device)
        distance = pos[:, None] - pos[None, :]
        scores = scores - self.distance_slopes[None, :, None, None] * distance.clamp(min=0)
        allowed = distance >= 0  # query i may see key j iff j <= i
        if window is not None:  # evaluation-only ablation: see just the last `window` tokens
            allowed = allowed & (distance < window)
        # Mask BEFORE softmax; forbidden positions have exp(-inf) = 0.
        weights = softmax(scores.masked_fill(~allowed, float("-inf")))
        mixed = weights @ v
        mixed = mixed.transpose(1, 2).contiguous().reshape(b, t, c)
        return self.out(mixed)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up = Linear(cfg.d_model, cfg.expansion * cfg.d_model)
        self.down = Linear(cfg.expansion * cfg.d_model, cfg.d_model)

    def forward(self, x):
        # MLP transforms each position independently; attention moves information.
        return self.down(gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg.d_model)
        self.attention = CausalAttention(cfg)
        self.norm2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, window=None):
        x = x + self.attention(self.norm1(x), window)  # residual: direct path for values/gradients
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Creation order fixes the RNG stream and parameter order -> do not reorder.
        self.token_embedding = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model) * 0.02)
        # Learned absolute positions. Rows beyond the training length get no data
        # gradient, so longer inputs are an extrapolation test, not a given.
        self.positions = nn.Parameter(torch.randn(cfg.max_seq_len, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.d_model)
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids, window=None):
        assert token_ids.shape[1] <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        # Integer lookup is differentiable with respect to selected embedding rows.
        x = self.token_embedding[token_ids] * math.sqrt(self.cfg.d_model)
        x = x + self.positions[:token_ids.shape[1]]
        for block in self.blocks:
            x = block(x, window)
        return self.lm_head(self.final_norm(x))  # [B,T,C] @ [C,V] -> [B,T,V]


def token_cross_entropy(logits, targets):
    """Return per-token NLL. logsumexp is stable even for very large logits."""
    correct = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.logsumexp(logits, dim=-1) - correct


def objective(logits, targets, answer_weight=1.0):
    nll = token_cross_entropy(logits, targets)
    lm_loss = nll.mean()
    answer_loss = nll[:, -1].mean()
    # A random filler token is unpredictable. Extra answer weight prevents its
    # many positions from drowning out the only long-range supervised signal.
    return lm_loss + answer_weight * answer_loss, lm_loss, answer_loss


class AdamW:
    """Explicit Adam moments, bias correction and decoupled weight decay."""
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(parameters)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        # Gradients accumulate by default; each minibatch needs a clean buffer.
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p.mul_(1 - self.lr * self.weight_decay)
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)


@torch.no_grad()
def clip_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    total = torch.stack([p.grad.square().sum() for p in params]).sum().sqrt()
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite gradient")
    scale = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for p in params:
        p.grad.mul_(scale)
    return total.item()
