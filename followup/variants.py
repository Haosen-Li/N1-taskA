"""Model / data variants built ON TOP of the unchanged taska code (nothing in taska/*.py is modified)."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # taska/
import model as M, task as T

ANSWER = 3  # vocab id 3 '<unused>' == the old-format '<answer>' marker

def build(cfg, noslope=False, pos="learned"):
    m = M.TransformerLM(cfg)            # same init RNG stream as the original
    if noslope:
        for b in m.blocks:
            b.attention.distance_slopes.zero_()
    if pos != "learned":
        del m.positions
        if pos == "none":
            pe = torch.zeros(cfg.max_seq_len, cfg.d_model)
        elif pos == "sinusoidal":
            p = torch.arange(cfg.max_seq_len)[:, None].float()
            i = torch.arange(0, cfg.d_model, 2).float()
            ang = p / 10000 ** (i / cfg.d_model)
            pe = torch.zeros(cfg.max_seq_len, cfg.d_model)
            pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
        else:
            raise ValueError(pos)
        m.register_buffer("positions", pe)  # fixed, not trained
    return m

def make(batch_size, gap, generator, device="cpu", offset=0, fmt="direct", offset_max=None):
    """offset = number of random filler tokens inserted between <bos> and the memory block.
    offset_max: training mode, offset ~ U{0..offset_max} per batch. fmt='old': <query> K <answer> -> V."""
    if offset_max is not None:
        offset = int(torch.randint(0, offset_max + 1, (1,), generator=generator))
    b = T.make_batch(batch_size, gap, generator, "cpu")
    seq = torch.cat([b.x, b.y[:, -1:]], 1)
    if fmt == "old":
        seq = torch.cat([seq[:, :-1], torch.full((batch_size, 1), ANSWER), seq[:, -1:]], 1)
    if offset:
        f = torch.randint(T.N_VALUES, (batch_size, offset), generator=generator) + T.VALUE_START
        seq = torch.cat([seq[:, :1], f, seq[:, 1:]], 1)
    return T.Batch(seq[:, :-1].to(device), seq[:, 1:].to(device),
                   (b.source_pos + offset).to(device), b.slot.to(device))
