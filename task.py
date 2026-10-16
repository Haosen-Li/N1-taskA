"""Online random associative recall; train/validation/test use independent RNG seeds.

Sequence:  <bos> (<store> K V)x4  [gap random filler values]  <query> K  ->  V
"""
from dataclasses import dataclass
import torch

BOS, STORE, QUERY = 0, 1, 2
N_KEYS = N_VALUES = 8
KEY_START, VALUE_START = 4, 12
# Token 3 is unused. It is kept so that the vocabulary size stays at 20, which
# keeps the parameter shapes and the init RNG stream the same as the trained checkpoint.
VOCAB = ["<bos>", "<store>", "<query>", "<unused>"]
VOCAB += [f"K{i}" for i in range(N_KEYS)] + [f"V{i}" for i in range(N_VALUES)]
TRAIN_GAPS = (32, 48, 64)
MEMORY_VALUE_POS = [3, 6, 9, 12]  # positions of the 4 stored values in x


def decode(tokens):
    return " ".join(VOCAB[int(i)] for i in tokens)


@dataclass
class Batch:
    x: torch.Tensor           # [B, T] model input
    y: torch.Tensor           # [B, T] next-token targets (x shifted by one)
    source_pos: torch.Tensor  # [B] position in x of the value that answers the query
    slot: torch.Tensor        # [B] which of the 4 records is queried

    @property
    def answer(self):
        return self.y[:, -1]


def make_batch(batch_size, gap, generator, device="cpu"):
    # Four DISTINCT random keys; values sampled independently with replacement.
    # Key identity, slot, length and filler therefore do not determine the answer.
    keys = torch.rand(batch_size, N_KEYS, generator=generator).argsort(-1)[:, :4] + KEY_START
    values = torch.randint(N_VALUES, (batch_size, 4), generator=generator) + VALUE_START
    slot = torch.randint(4, (batch_size,), generator=generator)
    row = torch.arange(batch_size)
    records = torch.empty(batch_size, 4, 3, dtype=torch.long)
    records[:, :, 0] = STORE
    records[:, :, 1] = keys
    records[:, :, 2] = values
    noise = torch.randint(N_VALUES, (batch_size, gap), generator=generator) + VALUE_START
    suffix = torch.stack([torch.full_like(slot, QUERY), keys[row, slot], values[row, slot]], dim=1)
    seq = torch.cat([torch.full((batch_size, 1), BOS), records.flatten(1), noise, suffix], dim=1)
    # Critical shift: the answer token is in y only, never in x.
    return Batch(seq[:, :-1].to(device), seq[:, 1:].to(device),
                 (3 + 3 * slot).to(device), slot.to(device))


def change_relevant_value(batch, offset=1):
    """Same suffix and all other tokens, but a different (remote) correct answer."""
    x = batch.x.clone()
    new = (batch.answer - VALUE_START + offset) % N_VALUES + VALUE_START
    x[torch.arange(len(x), device=x.device), batch.source_pos] = new
    return x, new


def change_irrelevant_value(batch, offset=1):
    """Change the value stored under a key that is NOT queried."""
    x = batch.x.clone()
    other_pos = 3 + 3 * ((batch.slot + 1) % 4)
    rows = torch.arange(len(x), device=x.device)
    old = x[rows, other_pos]
    x[rows, other_pos] = (old - VALUE_START + offset) % N_VALUES + VALUE_START
    return x
