#!/usr/bin/env python3
"""
RARE-JEPA on SCAN — compositional generalization benchmark.

Architecture: RAREJEPA encoder + small GRU autoregressive decoder.
  - Encoder: the 6 experts route over the input command
  - Decoder: GRU that takes pooled encoder state as init, generates actions
  - Teacher forcing during training, greedy decoding at eval
  - Metric: exact-match (full sequence must match target)

Reuses from rare_jepa.py: RAREJEPA model, training helpers, save_run, etc.

Splits: simple | length | addprim_jump | addprim_turn_left

Usage:
  python3 rare_scan.py --split addprim_jump --routing sequential
  python3 rare_scan.py --split addprim_jump --routing unified
"""
import sys, os, time, argparse, random
sys.path.insert(0, ".")
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rare_jepa as rj
import scan_loader

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RARE-JEPA on SCAN")
parser.add_argument("--split", type=str, default="simple",
                    choices=list(scan_loader.SCAN_URLS),
                    help="SCAN split: simple / length / addprim_jump / addprim_turn_left")
parser.add_argument("--routing", type=str, default="unified",
                    choices=["unified", "sequential"])
parser.add_argument("--total-ep", type=int, default=100)
parser.add_argument("--p1-end", type=int, default=20)
parser.add_argument("--tag", type=str, default=None,
                    help="Name tag (default: scan_<split>_<routing>)")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--max-action-len", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
if args.tag is None:
    args.tag = f"scan_{args.split}_{args.routing}"

SPLIT = args.split
ROUTING = args.routing
TOTAL_EP = args.total_ep
P1_END = args.p1_end
P2_LEN = TOTAL_EP - P1_END
BATCH = args.batch_size
MAX_ACT = args.max_action_len

print("=" * 70)
print(f"  RARE-JEPA on SCAN — split={SPLIT}  routing={ROUTING}")
print(f"  Phase 1 (0-{P1_END-1}) : forced sequential (encoder)")
print(f"  Phase 2 ({P1_END}-{TOTAL_EP-1}) : {ROUTING} routing, ε 0.3→0.05")
print(f"  Max action length: {MAX_ACT}")
print("=" * 70)

rj.seed_everything(args.seed)


# ── Vocab / Tokenizer ─────────────────────────────────────────
class Vocab:
    def __init__(self, extras=()):
        self.w2i = {"<pad>": 0, "<unk>": 1}
        self.i2w = {0: "<pad>", 1: "<unk>"}
        for e in extras:
            self.add(e)

    def add(self, w):
        if w not in self.w2i:
            i = len(self.w2i)
            self.w2i[w] = i
            self.i2w[i] = w

    def encode(self, tokens):
        return [self.w2i.get(t, 1) for t in tokens]

    def __len__(self):
        return len(self.w2i)


# ── Load SCAN ─────────────────────────────────────────────────
tr_pairs, te_pairs = scan_loader.load_split(SPLIT)
print(f"Loaded: {len(tr_pairs)} train / {len(te_pairs)} test")

# Build vocabs
in_vocab = Vocab()
out_vocab = Vocab(extras=["<bos>", "<eos>"])
for cmd, act in tr_pairs + te_pairs:
    for t in cmd.split(): in_vocab.add(t)
    for t in act.split(): out_vocab.add(t)
print(f"Input vocab: {len(in_vocab)}  Output vocab: {len(out_vocab)}")

BOS = out_vocab.w2i["<bos>"]
EOS = out_vocab.w2i["<eos>"]
PAD = 0


class SCANDataset(Dataset):
    def __init__(self, pairs, in_vocab, out_vocab, max_act=MAX_ACT):
        self.data = []
        for cmd, act in pairs:
            in_toks  = in_vocab.encode(cmd.split())
            out_toks = [BOS] + out_vocab.encode(act.split()) + [EOS]
            out_toks = out_toks[:max_act]
            self.data.append((in_toks, out_toks))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_scan(batch):
    inp, out = zip(*batch)
    in_lens  = [len(x) for x in inp]
    out_lens = [len(y) for y in out]
    Lin, Lout = max(in_lens), max(out_lens)
    B = len(batch)
    in_pad  = torch.zeros(B, Lin,  dtype=torch.long)
    out_pad = torch.zeros(B, Lout, dtype=torch.long)
    in_mask = torch.zeros(B, Lin,  dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        in_pad[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        out_pad[i, :len(y)] = torch.tensor(y, dtype=torch.long)
        in_mask[i, :len(x)] = True
    return in_pad, in_mask, out_pad, torch.tensor(out_lens)


# ── Model ─────────────────────────────────────────────────────
class SCANDecoder(nn.Module):
    """GRU-based autoregressive decoder with cross-attention to encoder seq."""

    def __init__(self, action_vocab_size, dim=None):
        super().__init__()
        if dim is None:
            dim = rj.HIDDEN_DIM
        self.dim = dim
        self.action_emb = nn.Embedding(action_vocab_size, dim)
        # Cross-attention to encoder tokens
        self.cross_attn = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=0.1)
        self.gru = nn.GRUCell(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, action_vocab_size)

    def step(self, prev_token_emb, hidden, enc_seq, enc_mask):
        # Cross-attention: query = hidden, key/value = enc_seq
        q = hidden.unsqueeze(1)                        # (B, 1, D)
        kpm = ~enc_mask                                # MHA mask: True=ignore
        ctx, _ = self.cross_attn(q.to(enc_seq.dtype), enc_seq, enc_seq,
                                 key_padding_mask=kpm)
        ctx = ctx.squeeze(1).to(hidden.dtype)          # (B, D)
        gru_input = torch.cat([prev_token_emb, ctx], dim=-1)
        hidden = self.gru(gru_input.float(), hidden)
        return hidden, self.out(self.norm(hidden))

    def forward_teacher_forced(self, init_hidden, target_actions,
                               enc_seq, enc_mask):
        """Teacher forcing: feed target[:-1], predict target[1:]."""
        B, T = target_actions.shape
        hidden = init_hidden
        logits_seq = []
        for t in range(T - 1):
            emb = self.action_emb(target_actions[:, t])
            hidden, logits = self.step(emb, hidden, enc_seq, enc_mask)
            logits_seq.append(logits)
        return torch.stack(logits_seq, dim=1)         # (B, T-1, V)

    def generate(self, init_hidden, enc_seq, enc_mask, max_len, bos_id, eos_id):
        """Greedy decoding."""
        B = init_hidden.size(0)
        hidden = init_hidden
        prev = torch.full((B,), bos_id, dtype=torch.long, device=init_hidden.device)
        done = torch.zeros(B, dtype=torch.bool, device=init_hidden.device)
        out_tokens = []
        for _ in range(max_len):
            emb = self.action_emb(prev)
            hidden, logits = self.step(emb, hidden, enc_seq, enc_mask)
            pred = logits.argmax(-1)
            pred = torch.where(done, torch.zeros_like(pred), pred)
            out_tokens.append(pred)
            done = done | (pred == eos_id)
            prev = pred
            if done.all():
                break
        return torch.stack(out_tokens, dim=1)


class SCANModel(nn.Module):
    """Encoder (RAREJEPA) + SCANDecoder."""
    def __init__(self, input_vocab_size, action_vocab_size,
                 routing="sequential", use_gru=True):
        super().__init__()
        self.encoder = rj.RAREJEPA(input_vocab_size, routing=routing,
                                    use_gru=use_gru)
        if routing == "unified":
            self.encoder.unified_submode = "jepa"
            self.encoder.action_mask_mode = "all"
        self.decoder = SCANDecoder(action_vocab_size)


# ── Training & Eval ───────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, phase, eps):
    model.train()
    tot_loss = 0.0
    tot_correct = 0
    tot_tokens = 0
    n_batches = 0

    for in_pad, in_mask, out_pad, out_lens in loader:
        in_pad   = in_pad.to(rj.DEVICE)
        in_mask  = in_mask.to(rj.DEVICE)
        out_pad  = out_pad.to(rj.DEVICE)

        if args.routing == "unified":
            model.encoder.unified_submode = "jepa"

        with rj._amp_autocast(rj.USE_AMP):
            # Encode
            _, enc_info = model.encoder(in_pad, in_mask, phase=phase,
                                          epsilon=eps, targets=None)
            enc_pooled = enc_info["pooled"]
            enc_seq    = enc_info["final_seq"]

            # Decode with teacher forcing
            dec_logits = model.decoder.forward_teacher_forced(
                enc_pooled, out_pad, enc_seq, in_mask)          # (B, T-1, V)
            targets = out_pad[:, 1:]                            # predict t>=1
            # Loss: ignore padding tokens
            mask = (targets != PAD).float()
            ce = F.cross_entropy(
                dec_logits.reshape(-1, dec_logits.size(-1)),
                targets.reshape(-1),
                reduction="none").view_as(targets)
            loss = (ce * mask).sum() / mask.sum().clamp(min=1)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if hasattr(model.encoder, "ema_experts"):
            with torch.no_grad():
                for p, pe in zip(model.encoder.experts.parameters(),
                                 model.encoder.ema_experts.parameters()):
                    pe.data.mul_(rj.EMA_DECAY).add_(p.data, alpha=1 - rj.EMA_DECAY)

        # Token accuracy
        with torch.no_grad():
            pred = dec_logits.argmax(-1)
            correct = ((pred == targets) & (targets != PAD)).sum().item()
            total = (targets != PAD).sum().item()
        tot_loss += loss.item()
        tot_correct += correct
        tot_tokens += total
        n_batches += 1

    return {
        "loss": tot_loss / max(n_batches, 1),
        "tok_acc": tot_correct / max(tot_tokens, 1) * 100,
    }


@torch.no_grad()
def evaluate(model, loader, max_len):
    model.eval()
    exact = 0
    tok_correct = 0
    tok_total = 0
    n = 0

    for in_pad, in_mask, out_pad, out_lens in loader:
        in_pad  = in_pad.to(rj.DEVICE)
        in_mask = in_mask.to(rj.DEVICE)
        out_pad = out_pad.to(rj.DEVICE)

        with rj._amp_autocast(rj.USE_AMP):
            _, enc_info = model.encoder(in_pad, in_mask, phase=2,
                                          epsilon=0.0, targets=None)
            enc_pooled = enc_info["pooled"]
            enc_seq    = enc_info["final_seq"]
            pred_seq = model.decoder.generate(
                enc_pooled, enc_seq, in_mask, max_len=max_len,
                bos_id=BOS, eos_id=EOS)                  # (B, L)

        # Target = out_pad[:, 1:] (skip BOS)
        tgt = out_pad[:, 1:]
        # Pad pred to tgt length
        L_tgt = tgt.size(1)
        L_pred = pred_seq.size(1)
        if L_pred < L_tgt:
            pad = torch.zeros(pred_seq.size(0), L_tgt - L_pred,
                              dtype=pred_seq.dtype, device=pred_seq.device)
            pred_seq = torch.cat([pred_seq, pad], dim=1)
        else:
            pred_seq = pred_seq[:, :L_tgt]

        # Exact match: all non-pad tokens match
        mask = (tgt != PAD)
        match = ((pred_seq == tgt) | ~mask).all(dim=1)
        exact += match.sum().item()

        tok_correct += ((pred_seq == tgt) & mask).sum().item()
        tok_total += mask.sum().item()
        n += tgt.size(0)

    return {
        "exact": exact / n * 100,
        "tok_acc": tok_correct / max(tok_total, 1) * 100,
    }


# ── Main run ──────────────────────────────────────────────────
tr_ds = SCANDataset(tr_pairs, in_vocab, out_vocab)
te_ds = SCANDataset(te_pairs, in_vocab, out_vocab)
tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, collate_fn=collate_scan,
                   num_workers=0, pin_memory=True)
te_ld = DataLoader(te_ds, BATCH, shuffle=False, collate_fn=collate_scan,
                   num_workers=0, pin_memory=True)

model = SCANModel(len(in_vocab), len(out_vocab),
                   routing=ROUTING, use_gru=True).to(rj.DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params (encoder routing={ROUTING})")

opt  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=rj.WEIGHT_DECAY)
sclr = rj._amp_scaler(rj.USE_AMP)

hist = defaultdict(list)
epoch_times = []
breakthrough_ep = None
t0 = time.time()

for ep in range(TOTAL_EP):
    if ep < P1_END:
        phase, eps = 1, 0.0
    else:
        progress = (ep - P1_END) / max(P2_LEN - 1, 1)
        eps = 0.3 - progress * (0.3 - 0.05)
        phase = 2

    t_ep = time.time()
    trm = train_epoch(model, tr_ld, opt, sclr, phase, eps)
    tem = evaluate(model, te_ld, max_len=MAX_ACT)
    dt = time.time() - t_ep
    epoch_times.append(dt)

    for k, v in trm.items(): hist[f"tr_{k}"].append(v)
    for k, v in tem.items(): hist[f"te_{k}"].append(v)

    if breakthrough_ep is None and tem["exact"] >= 95.0:
        breakthrough_ep = ep

    avg_ep = np.mean(epoch_times)
    remain = (TOTAL_EP - ep - 1) * avg_ep
    mark = "★" if breakthrough_ep == ep else " "
    print(f"E{ep:02d} {mark} p={phase} ε={eps:.2f} | loss {trm['loss']:.3f} "
          f"tok {trm['tok_acc']:.1f} | test: exact {tem['exact']:.1f}% "
          f"tok {tem['tok_acc']:.1f}% | {dt:.1f}s | "
          f"ETA {int(remain/60)}m{int(remain%60):02d}s")

# ── Summary ──────────────────────────────────────────────────
total_min = (time.time() - t0) / 60
print(f"\n{'=' * 70}")
print(f"SCAN {SPLIT} — {ROUTING}")
print(f"{'=' * 70}")
print(f"Final exact-match: {hist['te_exact'][-1]:.1f}%")
print(f"Max exact-match:   {max(hist['te_exact']):.1f}%  "
      f"(@ epoch {np.argmax(hist['te_exact'])})")
print(f"Breakthrough ≥95%: {'E' + str(breakthrough_ep) if breakthrough_ep is not None else '—'}")
print(f"Total time: {total_min:.1f} min")

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist["tr_tok_acc"], label="train token acc", color="#888", linewidth=1.2)
ax.plot(hist["te_tok_acc"], label="test token acc",  color="#3498db", linewidth=1.2)
ax.plot(hist["te_exact"],   label="test exact-match",color="#2ecc71", linewidth=2)
ax.axvline(P1_END, color="gray", ls="--", alpha=.4, label="Phase 2 start")
ax.axhline(95, color="red", ls=":", alpha=.3, label="95% threshold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy %")
ax.set_title(f"SCAN {SPLIT} — routing={ROUTING}")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(rj.PLOT_DIR, f"scan_{SPLIT}_{ROUTING}.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Plot: {out}")

fn = rj.save_run(args.tag, {
    "split": SPLIT,
    "routing": ROUTING,
    "total_ep": TOTAL_EP,
    "p1_end": P1_END,
    "hist": dict(hist),
    "final_exact": hist["te_exact"][-1],
    "max_exact": max(hist["te_exact"]),
    "breakthrough_ep": breakthrough_ep,
    "total_min": total_min,
    "epoch_times": epoch_times,
    "n_params": n_params,
}, subfolder="scan")
print(f"Run saved: {fn}")
