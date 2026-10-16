"""Numerical and information-flow tests, independent of successful training."""
import math
import pytest
import torch
import torch.nn.functional as F
from model import Config, TransformerLM, Block, CausalAttention, LayerNorm, AdamW, token_cross_entropy, clip_grad_norm
from task import make_batch, change_relevant_value, change_irrelevant_value, KEY_START, VALUE_START

torch.set_num_threads(2)


def test_loss_values_and_gradients_match_reference():
    torch.manual_seed(11)
    a = (torch.randn(2, 5, 20, dtype=torch.double) * 100).requires_grad_()
    b = a.detach().clone().requires_grad_()
    y = torch.randint(20, (2, 5))
    manual = token_cross_entropy(a, y).mean()
    reference = F.cross_entropy(b.flatten(0, 1), y.flatten())
    torch.testing.assert_close(manual, reference)
    manual.backward()
    reference.backward()
    torch.testing.assert_close(a.grad, b.grad)


def test_layer_norm_and_gradients_match_reference():
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    norm = LayerNorm(8).double()
    torch.testing.assert_close(norm(x), F.layer_norm(x, (8,), norm.scale, norm.bias, norm.eps))
    assert torch.autograd.gradcheck(norm, (x,), fast_mode=True)


def test_whole_block_finite_difference_gradient():
    torch.manual_seed(2)
    block = Block(Config(d_model=8, n_heads=2, n_layers=1)).double()
    x = torch.randn(1, 3, 8, dtype=torch.double, requires_grad=True)
    # Checks the Q/K/V, softmax, residual, LayerNorm and MLP chain w.r.t. input.
    assert torch.autograd.gradcheck(block, (x,), fast_mode=True)


def test_attention_matches_independent_reference():
    # Heads 0-1 have slope 0 (plain causal attention), heads 2-3 have a distance bias.
    cfg = Config(d_model=16, n_heads=4)
    layer = CausalAttention(cfg).double()
    x = torch.randn(2, 6, 16, dtype=torch.double)
    q, k, v = [(x @ layer.qkv.weight + layer.qkv.bias).chunk(3, -1)[i]
               .reshape(2, 6, 4, 4).transpose(1, 2) for i in range(3)]
    distance = torch.arange(6)[:, None] - torch.arange(6)[None, :]
    bias = -layer.distance_slopes[:, None, None] * distance.clamp(min=0)
    bias = bias.masked_fill(distance < 0, float("-inf"))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    ref = ref.transpose(1, 2).reshape(2, 6, 16) @ layer.out.weight + layer.out.bias
    torch.testing.assert_close(layer(x), ref)


def test_parameter_finite_difference_and_all_gradients():
    torch.manual_seed(3)
    model = TransformerLM(Config(d_model=8, n_heads=2, n_layers=1)).double()
    x = torch.tensor([[0, 1, 4, 12, 2, 4, 3]])
    y = torch.tensor([[1, 4, 12, 2, 4, 3, 12]])
    token_cross_entropy(model(x), y).mean().backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    eps = 1e-5
    for name, index in [("token_embedding", (4, 0)),
                        ("blocks.0.attention.qkv.weight", (0, 0)),
                        ("blocks.0.mlp.up.weight", (0, 0)),
                        ("lm_head.weight", (0, 12))]:
        p = dict(model.named_parameters())[name]
        expected = p.grad[index].item()
        original = p[index].item()
        with torch.no_grad():
            p[index] = original + eps
            plus = token_cross_entropy(model(x), y).mean().item()
            p[index] = original - eps
            minus = token_cross_entropy(model(x), y).mean().item()
            p[index] = original
        assert math.isclose(expected, (plus - minus) / (2 * eps), abs_tol=1e-7, rel_tol=1e-4)


def test_future_cannot_change_past_logits_or_gradients():
    torch.manual_seed(4)
    model = TransformerLM(Config(d_model=16, n_heads=2))
    a = torch.tensor([[0, 1, 4, 12, 2, 4, 3]])
    b = a.clone()
    b[:, 4:] = torch.tensor([17, 18, 19])
    torch.testing.assert_close(model(a)[:, :4], model(b)[:, :4], atol=0, rtol=0)
    model(b)[:, 3].sum().backward()
    assert model.token_embedding.grad[17:20].count_nonzero() == 0


@pytest.mark.parametrize("layers", [2, 3])
def test_local_receptive_field_cannot_relay_remote_answer(layers):
    model = TransformerLM(Config(d_model=16, n_heads=2, n_layers=layers))
    batch = make_batch(8, 32, torch.Generator().manual_seed(5))
    edited, _ = change_relevant_value(batch)
    # Each layer looks back <=7; stacking layers still cannot reach the memory.
    assert (batch.x.shape[1] - 1 - batch.source_pos).min() > layers * 7
    torch.testing.assert_close(model(batch.x, window=8)[:, -1],
                               model(edited, window=8)[:, -1], atol=0, rtol=0)


def test_adamw_matches_reference_across_steps():
    p = torch.nn.Parameter(torch.tensor([1.2, -0.7], dtype=torch.double))
    q = torch.nn.Parameter(p.detach().clone())
    ours = AdamW([p], lr=0.01, weight_decay=0.02)
    ref = torch.optim.AdamW([q], lr=0.01, weight_decay=0.02, foreach=False)
    for i in range(5):
        g = torch.tensor([0.1 + i, -0.3 - i * 0.1], dtype=torch.double)
        p.grad, q.grad = g.clone(), g.clone()
        ours.step()
        ref.step()
        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)
        ours.zero_grad()
        assert p.grad is None


def test_clipping_and_data_shift_and_interventions():
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([3.0, 4.0])
    assert clip_grad_norm([p], 1.0) == 5.0
    assert p.grad.norm() <= 1.0
    batch = make_batch(16, 32, torch.Generator().manual_seed(6))
    assert torch.equal(batch.x[:, 1:], batch.y[:, :-1])
    # Last INPUT is the query key; the answer value exists only in y.
    assert ((batch.x[:, -1] >= KEY_START) & (batch.x[:, -1] < VALUE_START)).all()
    assert torch.equal(batch.x[torch.arange(16), batch.source_pos], batch.answer)
    for offset in range(1, 8):
        edited, target = change_relevant_value(batch, offset)
        assert torch.equal(edited[:, 13:], batch.x[:, 13:])
        assert ((edited != batch.x).sum(1) == 1).all()
        assert (target != batch.answer).all()
    irrelevant = change_irrelevant_value(batch)
    assert torch.equal(irrelevant[torch.arange(16), batch.source_pos], batch.answer)


def test_counterfactual_metric_rejects_constant_guesser():
    from evaluate import exhaustive_counterfactuals

    class ConstantGuesser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            scores = torch.zeros(*x.shape, 20, device=x.device)
            scores[:, :, 12] = 1
            return scores

    result = exhaustive_counterfactuals(ConstantGuesser(), count=16, gap=32, seed=99)
    assert result["variant_accuracy"] == 0.125
    assert result["all_8_correct"]["accuracy"] == 0
