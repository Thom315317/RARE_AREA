#!/usr/bin/env python3
"""
SCAN addprim_jump — 4 variantes d'ablation décomposition/recomposition.

Standalone file. Ne dépend PAS du code JEPA-MoE.

Variantes :
  A0        : Transformer seq2seq baseline (point zéro, doit faire ~0%)
  A1        : A0 + supervision auxiliaire de rôle sur l'encoder
  A2        : A0 + Vector Quantization structurelle (K quantifié, V original)
  A2_nosup  : A2 sans L_commitment (ablation critique)

Usage:
  python3 scan_compositional.py --variant A0 --seed 42
  python3 scan_compositional.py --variant A1 --seed 42
  python3 scan_compositional.py --variant A2 --seed 42
  python3 scan_compositional.py --variant A2_nosup --seed 42

Recap CLI:
  python3 scan_compositional.py --recap runs/
"""
import os, sys, json, time, math, random, argparse, urllib.request, re
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scan_compositional")
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "scan_compositional")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt"
TEST_URL  = "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt"

SCAN_SPLIT_URLS = {
    "addprim_jump": (
        "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt",
        "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt",
    ),
    "addprim_turn_left": (
        "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_turn_left.txt",
        "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_turn_left.txt",
    ),
    "length": (
        "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt",
        "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt",
    ),
}

# Rôles structurels SCAN (dérivés de la grammaire)
ROLE_MAP = {
    "jump": "ACTION", "walk": "ACTION", "run": "ACTION", "look": "ACTION",
    "left": "DIRECTION", "right": "DIRECTION",
    "opposite": "MOD_DIR", "around": "MOD_DIR",
    "twice": "QUANTIFIER", "thrice": "QUANTIFIER",
    "and": "CONNECTOR", "after": "CONNECTOR",
    "turn": "TURN",
}
ROLE_NAMES = ["ACTION", "DIRECTION", "MOD_DIR", "QUANTIFIER", "CONNECTOR", "TURN", "UNK"]
ROLE_TO_ID = {r: i for i, r in enumerate(ROLE_NAMES)}

# Tokens spéciaux pour l'output vocabulary
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


# ══════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════
def _download(url: str, dest: str):
    if not os.path.exists(dest):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)


def parse_scan_file(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            m = re.match(r"IN:\s*(.*?)\s*OUT:\s*(.*)$", line.strip())
            if m:
                pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


def build_vocabs(pairs):
    in_w2i = {PAD: 0}
    out_w2i = {PAD: 0, BOS: 1, EOS: 2}
    for cmd, act in pairs:
        for t in cmd.split():
            if t not in in_w2i: in_w2i[t] = len(in_w2i)
        for t in act.split():
            if t not in out_w2i: out_w2i[t] = len(out_w2i)
    return in_w2i, out_w2i


def split_train_val(pairs, val_frac=0.1, seed=42):
    """Stratified split by command length."""
    rng = random.Random(seed)
    by_len: Dict[int, list] = {}
    for p in pairs:
        by_len.setdefault(len(p[0].split()), []).append(p)
    train, val = [], []
    for L, group in by_len.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_frac))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    return train, val


class SCANDataset(Dataset):
    """SCAN dataset with optional action permutation.
    perm_in/perm_out: lists of tokens to permute (configurable per variant)."""

    def __init__(self, pairs, in_w2i, out_w2i, max_out=60,
                 permute_actions=False, perm_in=None, perm_out=None):
        self.data = []
        for cmd, act in pairs:
            src = [in_w2i.get(t, 0) for t in cmd.split()]
            tgt = [out_w2i[BOS]] + [out_w2i[t] for t in act.split()] + [out_w2i[EOS]]
            tgt = tgt[:max_out]
            roles = [ROLE_TO_ID.get(ROLE_MAP.get(t, "UNK"), ROLE_TO_ID["UNK"])
                     for t in cmd.split()]
            self.data.append((src, tgt, roles, len(cmd.split())))
        self.permute = permute_actions
        if perm_in is None:
            perm_in  = ["walk", "run", "look", "jump"]
        if perm_out is None:
            perm_out = ["I_WALK", "I_RUN", "I_LOOK", "I_JUMP"]
        self.perm_in_ids  = [in_w2i[t]  for t in perm_in  if t in in_w2i]
        self.perm_out_ids = [out_w2i[t] for t in perm_out if t in out_w2i]
        self.perm_n = len(perm_in)
        if self.permute and (len(self.perm_in_ids) != self.perm_n
                              or len(self.perm_out_ids) != self.perm_n):
            raise RuntimeError(
                f"Permutation requires all of {perm_in}/{perm_out} "
                f"in vocab (got {len(self.perm_in_ids)}/{len(self.perm_out_ids)})")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        src, tgt, roles, n = self.data[i]
        if self.permute:
            K = self.perm_n
            order = list(range(K))
            random.shuffle(order)
            m_in  = {self.perm_in_ids[k]:  self.perm_in_ids[order[k]]  for k in range(K)}
            m_out = {self.perm_out_ids[k]: self.perm_out_ids[order[k]] for k in range(K)}
            src = [m_in.get(t, t)  for t in src]
            tgt = [m_out.get(t, t) for t in tgt]
        return (src, tgt, roles, n)


def collate(batch):
    src, tgt, roles, cmd_lens = zip(*batch)
    S = max(len(x) for x in src)
    T = max(len(x) for x in tgt)
    B = len(batch)
    s = torch.zeros(B, S, dtype=torch.long)
    t = torch.zeros(B, T, dtype=torch.long)
    r = torch.zeros(B, S, dtype=torch.long)
    s_mask = torch.zeros(B, S, dtype=torch.bool)
    for i, (a, b, c, _) in enumerate(batch):
        s[i, :len(a)] = torch.tensor(a)
        t[i, :len(b)] = torch.tensor(b)
        r[i, :len(c)] = torch.tensor(c)
        s_mask[i, :len(a)] = True
    return s, s_mask, t, r, torch.tensor(cmd_lens)


# ══════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════
class CrossAttnDecoderLayer(nn.Module):
    """Pre-LN decoder block with separate K/V in cross-attention (for A2)."""

    def __init__(self, d_model, n_heads, ffn, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem_k, mem_v, causal_mask, mem_kpm):
        # Self-attention (causal)
        h = self.n1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(h)
        # Cross-attention with separated K/V
        h = self.n2(x)
        h, _ = self.cross_attn(h, mem_k, mem_v, key_padding_mask=mem_kpm, need_weights=False)
        x = x + self.drop(h)
        # FFN
        h = self.n3(x)
        x = x + self.drop(self.ffn(h))
        return x


class VQBottleneck(nn.Module):
    """Vector Quantization with Gumbel-Softmax straight-through."""

    def __init__(self, d_model, role_dim=32, n_codes=6):
        super().__init__()
        self.proj_in = nn.Linear(d_model, role_dim)
        self.codebook = nn.Parameter(torch.randn(n_codes, role_dim) * 0.5)
        self.proj_out = nn.Linear(role_dim, d_model)
        self.n_codes = n_codes
        self.role_dim = role_dim

    def forward(self, x, tau=1.0, hard=True):
        # x: (B, L, D)
        z = self.proj_in(x)                                    # (B, L, R)
        # Similarité à chaque code (negative distance)
        # (B, L, R) vs (n, R) → (B, L, n)
        logits = -((z.unsqueeze(-2) - self.codebook) ** 2).sum(-1)
        if self.training:
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard)  # (B, L, n)
        else:
            probs = F.one_hot(logits.argmax(-1), self.n_codes).to(logits.dtype)
        quantized_role = probs @ self.codebook                  # (B, L, R)
        out = self.proj_out(quantized_role)                     # (B, L, D)
        return out, probs


# ══════════════════════════════════════════════════════════════
# Base model + variants
# ══════════════════════════════════════════════════════════════
class BaseSeq2Seq(nn.Module):
    """Encoder Transformer + custom decoder (Pre-LN). Parent class."""

    def __init__(self, in_vocab, out_vocab, d_model=128, n_heads=4,
                 n_layers=2, ffn=256, dropout=0.1, max_in=20, max_out=60):
        super().__init__()
        self.d_model = d_model
        self.in_emb  = nn.Embedding(in_vocab, d_model, padding_idx=0)
        self.out_emb = nn.Embedding(out_vocab, d_model, padding_idx=0)
        self.in_pe   = nn.Embedding(max_in, d_model)
        self.out_pe  = nn.Embedding(max_out, d_model)
        # Encoder via torch standard
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn, dropout, activation=F.gelu,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        # Decoder custom (pour contrôle K/V en cross-attention)
        self.dec_layers = nn.ModuleList([
            CrossAttnDecoderLayer(d_model, n_heads, ffn, dropout)
            for _ in range(n_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, out_vocab)

    def encode(self, src, src_mask):
        pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        x = self.in_emb(src) + self.in_pe(pos)
        return self.encoder(x, src_key_padding_mask=~src_mask)

    def decode(self, memory_k, memory_v, mem_kpm, tgt):
        # tgt: (B, T)
        B, T = tgt.shape
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        x = self.out_emb(tgt) + self.out_pe(pos)
        causal = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), 1)
        for layer in self.dec_layers:
            x = layer(x, memory_k, memory_v, causal, mem_kpm)
        x = self.dec_norm(x)
        return self.out_head(x)                                  # (B, T, V)

    def get_memory_kv(self, enc_out):
        """Override in subclasses for different K/V strategies."""
        return enc_out, enc_out

    def forward(self, src, src_mask, tgt_in):
        """Teacher-forcing forward. Returns logits + aux dict."""
        enc_out = self.encode(src, src_mask)
        mem_k, mem_v = self.get_memory_kv(enc_out)
        logits = self.decode(mem_k, mem_v, ~src_mask, tgt_in)
        return logits, {"enc_out": enc_out}


class A0_Baseline(BaseSeq2Seq):
    """Standard seq2seq — memory_k = memory_v = encoder_out."""
    pass


class A4_permute(A0_Baseline):
    """Same architecture as A0. Only training-time data augmentation :
    per-example random permutation of (walk, run, look) tokens,
    consistently applied in input AND output."""
    pass


class A1_RoleSuper(BaseSeq2Seq):
    """A0 + auxiliary role prediction head on encoder tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role_head = nn.Linear(self.d_model, len(ROLE_NAMES))

    def forward(self, src, src_mask, tgt_in):
        enc_out = self.encode(src, src_mask)
        mem_k, mem_v = self.get_memory_kv(enc_out)
        logits = self.decode(mem_k, mem_v, ~src_mask, tgt_in)
        role_logits = self.role_head(enc_out)                    # (B, L, n_roles)
        return logits, {"enc_out": enc_out, "role_logits": role_logits}


class A2_VQ(BaseSeq2Seq):
    """A0 + VQ bottleneck. memory_k = quantized, memory_v = original."""

    def __init__(self, *args, n_codes=6, role_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.vq = VQBottleneck(self.d_model, role_dim, n_codes)
        self.tau = 2.0  # annealed externally

    def forward(self, src, src_mask, tgt_in):
        enc_out = self.encode(src, src_mask)
        q_out, q_probs = self.vq(enc_out, tau=self.tau, hard=True)
        # Cross-attn : K = quantized, V = original
        logits = self.decode(q_out, enc_out, ~src_mask, tgt_in)
        return logits, {"enc_out": enc_out, "q_out": q_out, "q_probs": q_probs}


class A2_nosup(A2_VQ):
    """A2 sans L_commitment. Seul L_action drive le VQ."""
    pass


class A2_warm(A2_VQ):
    """A2 avec codebook warm-started depuis un checkpoint A1 (centroids par rôle).
    Tau fixé à 2.0 pendant 50 epochs, puis decay linéaire vers 0.5."""
    pass


class A3_dual(BaseSeq2Seq):
    """Dual encoder : Structure (avec consistency loss) + Content (task only).
    Decoder cross-attention : Q = dec state, K = struct_out, V = content_out.

    L_consist : pour chaque batch, on construit une version augmentée où les
    primitives d'action (jump/walk/run/look) sont remplacées aléatoirement.
    L'encoder Structure doit produire des sorties IDENTIQUES aux positions
    non-action entre original et augmenté. L'encoder Content n'est PAS impacté.
    """

    ACTION_TOKENS = ["jump", "walk", "run", "look"]

    def __init__(self, in_vocab, out_vocab, d_model=128, n_heads=4,
                 n_layers=2, ffn=256, dropout=0.1, max_in=20, max_out=60):
        super().__init__(in_vocab, out_vocab, d_model, n_heads, n_layers,
                          ffn, dropout, max_in, max_out)
        # self.encoder (parent) = Structure encoder
        # Crée l'encoder Content séparé (mêmes hyperparams)
        enc_layer_c = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn, dropout, activation=F.gelu,
            batch_first=True, norm_first=True)
        self.enc_content = nn.TransformerEncoder(enc_layer_c, n_layers)
        # Placeholder for action_ids (set via set_action_ids)
        self.register_buffer("action_ids_buf",
                             torch.zeros(len(self.ACTION_TOKENS), dtype=torch.long))
        self._action_ids_set = False

    def set_action_ids(self, in_w2i):
        """Call after construction with the input vocab dict."""
        ids = [in_w2i[t] for t in self.ACTION_TOKENS if t in in_w2i]
        self.action_ids_buf = torch.tensor(ids, dtype=torch.long,
                                            device=self.action_ids_buf.device)
        self._action_ids_set = True

    def encode_structure(self, src, src_mask):
        pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        x = self.in_emb(src) + self.in_pe(pos)
        return self.encoder(x, src_key_padding_mask=~src_mask)

    def encode_content(self, src, src_mask):
        pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        x = self.in_emb(src) + self.in_pe(pos)
        return self.enc_content(x, src_key_padding_mask=~src_mask)

    def augment_actions(self, src):
        """Replace each action token at random with another random action token."""
        if not self._action_ids_set:
            raise RuntimeError("Call set_action_ids(in_w2i) after building A3_dual.")
        aid = self.action_ids_buf.to(src.device)
        is_action = torch.zeros_like(src, dtype=torch.bool)
        for a in aid:
            is_action = is_action | (src == a)
        rand_idx = torch.randint(0, aid.size(0), src.shape, device=src.device)
        rand_tokens = aid[rand_idx]
        src_aug = torch.where(is_action, rand_tokens, src)
        return src_aug, is_action

    def forward(self, src, src_mask, tgt_in):
        struct_out  = self.encode_structure(src, src_mask)              # (B, L, D)
        content_out = self.encode_content(src, src_mask)                # (B, L, D)
        # Cross-attention: K = struct, V = content
        logits = self.decode(struct_out, content_out, ~src_mask, tgt_in)

        info = {"struct_out": struct_out, "content_out": content_out}
        if self.training:
            # Compute augmented version for consistency loss
            src_aug, is_action = self.augment_actions(src)
            struct_aug = self.encode_structure(src_aug, src_mask)
            info["struct_aug"]  = struct_aug
            info["is_action"]   = is_action
            info["src_mask"]    = src_mask
        return logits, info


class A2_pre(A2_VQ):
    """VQ sur les embeddings d'entrée AVANT l'encoder.
    - Token emb + PE → VQ (6 prototypes, Gumbel-ST) → encoder
    - Cross-attention decoder : K = enc_out (factorisé), V = raw_emb (identité)
    - Tau fixé 50 ep puis decay 2.0→0.5.
    """

    def forward(self, src, src_mask, tgt_in):
        B, L = src.shape
        pos = torch.arange(L, device=src.device).unsqueeze(0)
        raw_emb = self.in_emb(src) + self.in_pe(pos)               # (B, L, D) — identité
        # Quantification AU NIVEAU DES EMBEDDINGS D'ENTRÉE
        q_emb, q_probs = self.vq(raw_emb, tau=self.tau, hard=True) # (B, L, D)
        # Encoder sur les embeddings quantifiés
        enc_out = self.encoder(q_emb, src_key_padding_mask=~src_mask)  # (B, L, D)
        # Cross-attention : K = enc_out (structure factorisée), V = raw_emb (identité bypass)
        logits = self.decode(enc_out, raw_emb, ~src_mask, tgt_in)
        return logits, {"enc_out": enc_out, "raw_emb": raw_emb,
                        "q_out": q_emb, "q_probs": q_probs}


def warmstart_codebook_from_a1(a2_model, a1_ckpt_path, train_loader, in_vocab,
                                 out_vocab, d_model, device):
    """Charge A1 depuis checkpoint, passe le train, calcule 6 centroids par rôle,
    les projette via A2's proj_in et remplace le codebook VQ."""
    print(f"Warm-start: loading A1 checkpoint from {a1_ckpt_path}")
    ckpt = torch.load(a1_ckpt_path, map_location=device, weights_only=False)

    # Rebuild A1 architecture to load weights
    a1 = A1_RoleSuper(in_vocab, out_vocab, d_model=d_model).to(device)
    a1.load_state_dict(ckpt["state_dict"])
    a1.eval()

    n_codes = a2_model.vq.n_codes
    sums = torch.zeros(n_codes, d_model, device=device)
    counts = torch.zeros(n_codes, device=device)

    with torch.no_grad():
        for src, src_mask, _, roles, _ in train_loader:
            src = src.to(device); src_mask = src_mask.to(device); roles = roles.to(device)
            enc = a1.encode(src, src_mask)                   # (B, L, D)
            # Only use tokens that are real (mask) AND role in [0, n_codes)
            for r in range(n_codes):
                m = (roles == r) & src_mask
                if m.any():
                    vecs = enc[m]                            # (N, D)
                    sums[r] += vecs.sum(0)
                    counts[r] += m.sum().float()

    # Fallback: if a role has 0 samples, use random init
    centroids_d = torch.zeros(n_codes, d_model, device=device)
    for r in range(n_codes):
        if counts[r] > 0:
            centroids_d[r] = sums[r] / counts[r]
        else:
            centroids_d[r] = torch.randn(d_model, device=device) * 0.5
            print(f"  WARNING: role {r} ({ROLE_NAMES[r]}) has 0 samples, random init")

    # Project through A2's proj_in (random init) to role_dim
    with torch.no_grad():
        centroids_role = a2_model.vq.proj_in(centroids_d)    # (n_codes, role_dim)

    # Replace codebook
    a2_model.vq.codebook.data.copy_(centroids_role)
    print(f"  Codebook warm-started: {n_codes} prototypes from A1 role clusters")
    print(f"  Counts per role: {counts.tolist()}")
    del a1
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# Training + evaluation
# ══════════════════════════════════════════════════════════════
def make_model(variant, in_vocab, out_vocab, d_model=128):
    if variant == "A0":
        return A0_Baseline(in_vocab, out_vocab, d_model=d_model)
    if variant == "A1":
        return A1_RoleSuper(in_vocab, out_vocab, d_model=d_model)
    if variant == "A2":
        return A2_VQ(in_vocab, out_vocab, d_model=d_model)
    if variant == "A2_nosup":
        return A2_nosup(in_vocab, out_vocab, d_model=d_model)
    if variant == "A2_warm":
        return A2_warm(in_vocab, out_vocab, d_model=d_model)
    if variant == "A2_pre":
        return A2_pre(in_vocab, out_vocab, d_model=d_model)
    if variant == "A3_dual":
        return A3_dual(in_vocab, out_vocab, d_model=d_model)
    if variant in ("A4_permute", "A4_seen", "A4_all"):
        return A4_permute(in_vocab, out_vocab, d_model=d_model)
    raise ValueError(f"Unknown variant: {variant}")


def compute_loss(variant, logits, tgt_out, info, args, targets_roles, src_mask):
    """Returns total loss + parts dict."""
    # Action CE with label smoothing
    V = logits.size(-1)
    flat_logits = logits.reshape(-1, V)
    flat_tgt = tgt_out.reshape(-1)
    # Ignore padding
    L_action = F.cross_entropy(flat_logits, flat_tgt, ignore_index=0,
                                label_smoothing=0.1)
    parts = {"L_action": L_action.item()}
    total = L_action

    if variant == "A1":
        # Role supervision on encoder tokens (ignore padding)
        role_logits = info["role_logits"]                       # (B, L, R)
        role_logits_flat = role_logits.reshape(-1, role_logits.size(-1))
        tgt_roles_flat = targets_roles.reshape(-1)
        # Mask padded positions
        mask_flat = src_mask.reshape(-1)
        L_role = F.cross_entropy(role_logits_flat[mask_flat],
                                 tgt_roles_flat[mask_flat])
        total = L_action + args.lambda_role * L_role
        parts["L_role"] = L_role.item()

    elif variant in ("A2", "A2_warm", "A2_pre"):
        # Commitment + codebook entropy
        enc_out = info["enc_out"]
        q_out   = info["q_out"]
        q_probs = info["q_probs"]                               # (B, L, K)
        L_commit = F.mse_loss(enc_out, q_out.detach())
        avg_p = q_probs.mean(dim=(0, 1))
        L_ent = -(-(avg_p * (avg_p + 1e-10).log()).sum())       # minimise → maximise entropy
        total = L_action + args.lambda_commit * L_commit + args.lambda_ent * L_ent
        parts["L_commit"] = L_commit.item()
        parts["L_ent"] = L_ent.item()

    elif variant == "A2_nosup":
        # No aux losses — pure action gradient drives VQ
        pass

    elif variant == "A3_dual":
        # Consistency loss on Structure encoder at non-action positions
        struct_out = info["struct_out"]                          # (B, L, D)
        struct_aug = info["struct_aug"]                          # (B, L, D)
        is_action  = info["is_action"]                           # (B, L)
        mask       = info["src_mask"]                            # (B, L)
        non_action_mask = (~is_action) & mask                    # (B, L)
        diff = (struct_out - struct_aug) ** 2                    # (B, L, D)
        m = non_action_mask.unsqueeze(-1).float()
        denom = m.sum().clamp(min=1.0) * struct_out.size(-1)
        L_consist = (diff * m).sum() / denom
        total = L_action + args.lambda_consist * L_consist
        parts["L_consist"] = L_consist.item()

    return total, parts


@torch.no_grad()
def greedy_decode(model, src, src_mask, out_w2i, max_len=50):
    """Greedy auto-regressive decoding."""
    B = src.size(0)
    device = src.device
    bos = out_w2i[BOS]
    eos = out_w2i[EOS]
    # Encode once (variant-specific)
    if isinstance(model, A2_pre):
        # VQ AVANT l'encoder. K = enc(quantized), V = raw_emb
        B, L = src.shape
        pos = torch.arange(L, device=src.device).unsqueeze(0)
        raw_emb = model.in_emb(src) + model.in_pe(pos)
        q_emb, _ = model.vq(raw_emb, tau=model.tau, hard=True)
        enc_out = model.encoder(q_emb, src_key_padding_mask=~src_mask)
        mem_k, mem_v = enc_out, raw_emb
    elif isinstance(model, A2_VQ):
        enc_out = model.encode(src, src_mask)
        q_out, _ = model.vq(enc_out, tau=model.tau, hard=True)
        mem_k, mem_v = q_out, enc_out       # K quantized, V original (A2 classique)
    elif isinstance(model, A3_dual):
        struct_out  = model.encode_structure(src, src_mask)
        content_out = model.encode_content(src, src_mask)
        mem_k, mem_v = struct_out, content_out   # K struct, V content
    else:
        enc_out = model.encode(src, src_mask)
        mem_k, mem_v = enc_out, enc_out
    mem_kpm = ~src_mask

    tgt = torch.full((B, 1), bos, device=device, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits = model.decode(mem_k, mem_v, mem_kpm, tgt)       # (B, t, V)
        nxt = logits[:, -1].argmax(-1)                          # (B,)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == eos)
        if done.all(): break
    return tgt[:, 1:]                                            # skip BOS


@torch.no_grad()
def evaluate(model, loader, out_w2i, device, max_len=50):
    model.eval()
    n, exact, tok_c, tok_t = 0, 0, 0, 0
    by_len: Dict[int, Tuple[int, int]] = {}   # cmd_len → (total, correct)

    for src, src_mask, tgt, _, cmd_lens in loader:
        src = src.to(device); src_mask = src_mask.to(device); tgt = tgt.to(device)
        pred = greedy_decode(model, src, src_mask, out_w2i, max_len=max_len)
        # target for comparison = tgt[:, 1:] (skip BOS)
        tgt_cmp = tgt[:, 1:]
        # Pad pred to tgt_cmp length
        Lp, Lt = pred.size(1), tgt_cmp.size(1)
        if Lp < Lt:
            pad = torch.zeros(pred.size(0), Lt - Lp, dtype=pred.dtype, device=device)
            pred = torch.cat([pred, pad], dim=1)
        else:
            pred = pred[:, :Lt]
        mask = (tgt_cmp != 0)
        match = ((pred == tgt_cmp) | ~mask).all(dim=1)
        exact += match.sum().item()
        tok_c += ((pred == tgt_cmp) & mask).sum().item()
        tok_t += mask.sum().item()
        n += src.size(0)
        for i, L in enumerate(cmd_lens.tolist()):
            t_, c_ = by_len.get(L, (0, 0))
            by_len[L] = (t_ + 1, c_ + (1 if match[i].item() else 0))

    return {
        "exact": exact / n * 100,
        "tok_acc": tok_c / max(tok_t, 1) * 100,
        "by_cmd_length": {L: (c / t * 100) for L, (t, c) in by_len.items()},
    }


def train_one_run(variant: str, seed: int, args):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.runs_dir if args.runs_dir else RUNS_DIR
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, f"{variant}_s{seed}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  Variant {variant}  |  seed {seed}  |  dir={run_dir}")
    print(f"{'='*70}")

    # Data — resolve split URLs
    split = getattr(args, "split", "addprim_jump")
    train_url, test_url = SCAN_SPLIT_URLS.get(split, (TRAIN_URL, TEST_URL))
    train_path = os.path.join(DATA_DIR, f"train_{split}.txt")
    test_path  = os.path.join(DATA_DIR, f"test_{split}.txt")
    _download(train_url, train_path)
    _download(test_url, test_path)
    print(f"  Split: {split}")
    train_pairs_full = parse_scan_file(train_path)
    test_pairs  = parse_scan_file(test_path)
    train_pairs, val_pairs = split_train_val(train_pairs_full, 0.1, seed=seed)

    in_w2i, out_w2i = build_vocabs(train_pairs_full + test_pairs)
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}  Test: {len(test_pairs)}")
    print(f"Input vocab: {len(in_w2i)}  Output vocab: {len(out_w2i)}")

    # Configure permutation pool based on variant + split
    do_perm = variant in ("A4_permute", "A4_seen", "A4_all")
    perm_in, perm_out = None, None   # defaults: all 4 actions
    if variant == "A4_seen":
        # Permute only primitives already seen in composition (exclude the held-out one)
        SEEN_POOLS = {
            "addprim_jump":      (["walk", "run", "look"],         ["I_WALK", "I_RUN", "I_LOOK"]),
            "addprim_turn_left": (["walk", "run", "look", "jump"], ["I_WALK", "I_RUN", "I_LOOK", "I_JUMP"]),
        }
        perm_in, perm_out = SEEN_POOLS.get(split, (None, None))
        if perm_in is None:
            print(f"  WARNING: A4_seen not defined for split {split}, skipping permutation")
            do_perm = False
        else:
            print(f"  A4_seen pool: {perm_in}")
    elif variant == "A4_all":
        ALL_POOLS = {
            "addprim_jump":      (["walk", "run", "look", "jump"], ["I_WALK", "I_RUN", "I_LOOK", "I_JUMP"]),
            "addprim_turn_left": (["walk", "run", "look", "jump"], ["I_WALK", "I_RUN", "I_LOOK", "I_JUMP"]),
            # turn_left bigram not supported yet — use same 4 actions
        }
        perm_in, perm_out = ALL_POOLS.get(split, (None, None))
        if perm_in is None:
            print(f"  WARNING: A4_all not defined for split {split}, skipping permutation")
            do_perm = False
        else:
            print(f"  A4_all pool: {perm_in}")
    elif variant == "A4_permute":
        print(f"  A4_permute pool: [walk, run, look, jump] (legacy)")

    tr_ds = SCANDataset(train_pairs, in_w2i, out_w2i,
                        permute_actions=do_perm, perm_in=perm_in, perm_out=perm_out)
    va_ds = SCANDataset(val_pairs,   in_w2i, out_w2i)
    te_ds = SCANDataset(test_pairs,  in_w2i, out_w2i)
    tr_ld = DataLoader(tr_ds, args.batch_size, shuffle=True,  collate_fn=collate, num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, args.batch_size, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, args.batch_size, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)

    # Model
    model = make_model(variant, len(in_w2i), len(out_w2i), d_model=args.d_model).to(device)
    if isinstance(model, A3_dual):
        model.set_action_ids(in_w2i)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M")

    # A2_warm: initialize VQ codebook from A1 role centroids
    if variant == "A2_warm":
        if not args.a1_checkpoint:
            raise ValueError("A2_warm requires --a1-checkpoint PATH")
        if not os.path.exists(args.a1_checkpoint):
            raise FileNotFoundError(f"A1 checkpoint not found: {args.a1_checkpoint}")
        warmstart_codebook_from_a1(
            model, args.a1_checkpoint, tr_ld,
            len(in_w2i), len(out_w2i), args.d_model, device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    # LR schedule: warmup + cosine
    total_steps = args.epochs * len(tr_ld)
    warmup = min(500, total_steps // 10)
    def lr_at(step):
        if step < warmup: return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    metrics_log = []
    best_val_exact = -1.0
    patience = 0
    step = 0
    epoch_times = []

    for ep in range(args.epochs):
        t_ep = time.time()
        # Tau annealing for A2 / A2_nosup
        if isinstance(model, A2_VQ):
            if isinstance(model, (A2_warm, A2_pre)):
                # Tau fixé à 2.0 pendant 50 ep, puis decay linéaire vers 0.5
                warm_hold = args.warm_hold_epochs
                if ep < warm_hold:
                    model.tau = 2.0
                else:
                    remain = max(args.epochs - warm_hold - 1, 1)
                    progress = min((ep - warm_hold) / remain, 1.0)
                    model.tau = 2.0 - progress * 1.5      # 2.0 → 0.5
            else:
                # A2 / A2_nosup: decay linéaire sur la première moitié
                progress = min(ep / max(args.epochs * 0.5, 1), 1.0)
                model.tau = 2.0 - progress * 1.5         # 2.0 → 0.5

        # Training
        model.train()
        tr_loss = 0.0; n_batches = 0
        loss_parts_sum = {}
        for src, src_mask, tgt, roles, _ in tr_ld:
            src = src.to(device); src_mask = src_mask.to(device); tgt = tgt.to(device); roles = roles.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, info = model(src, src_mask, tgt_in)
                loss, parts = compute_loss(variant, logits, tgt_out, info, args, roles, src_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_loss += loss.item(); n_batches += 1
            for k, v in parts.items():
                loss_parts_sum[k] = loss_parts_sum.get(k, 0.0) + v
            step += 1

        # Evaluation
        val_metrics  = evaluate(model, va_ld, out_w2i, device)
        test_metrics = evaluate(model, te_ld, out_w2i, device)

        # VQ stats if applicable
        vq_stats = {}
        if isinstance(model, A2_VQ):
            with torch.no_grad():
                # Collect prototype usage on one training batch
                for src, src_mask, _, _, _ in tr_ld:
                    src = src.to(device); src_mask = src_mask.to(device)
                    enc_out = model.encode(src, src_mask)
                    _, q_probs = model.vq(enc_out, tau=model.tau, hard=True)
                    mask_flat = src_mask.flatten()
                    assigns = q_probs.argmax(-1).flatten()[mask_flat]
                    counts = torch.bincount(assigns, minlength=model.vq.n_codes).float()
                    counts /= counts.sum().clamp(min=1)
                    entropy = -(counts * (counts + 1e-10).log()).sum().item()
                    vq_stats = {
                        "proto_usage": counts.cpu().tolist(),
                        "proto_entropy": entropy,
                        "tau": model.tau,
                    }
                    # Matrix: token word → prototype (every 10 epochs)
                    if ep % 10 == 0:
                        token_proto = {}
                        for b in range(src.size(0)):
                            for l in range(src.size(1)):
                                if not src_mask[b, l].item(): continue
                                w = src[b, l].item()
                                p = q_probs[b, l].argmax().item()
                                key = f"id_{w}"
                                token_proto.setdefault(key, [0] * model.vq.n_codes)
                                token_proto[key][p] += 1
                        vq_stats["token_to_proto_sample"] = token_proto
                    break  # one batch

        # Log
        entry = {
            "epoch": ep,
            "train_loss": tr_loss / max(n_batches, 1),
            "train_loss_parts": {k: v / max(n_batches, 1) for k, v in loss_parts_sum.items()},
            "val": val_metrics,
            "test": test_metrics,
            "lr": sched.get_last_lr()[0],
        }
        if vq_stats: entry["vq"] = vq_stats
        metrics_log.append(entry)

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics_log, f, indent=1)

        is_best = val_metrics["exact"] > best_val_exact
        if is_best:
            best_val_exact = val_metrics["exact"]
            patience = 0
        else:
            patience += 1

        dt = time.time() - t_ep
        epoch_times.append(dt)
        avg_ep = sum(epoch_times) / len(epoch_times)
        # Pessimistic ETA: assume (args.epochs - ep - 1) more epochs OR
        # early-stop when patience exhausted, whichever first
        remaining_ep = min(args.epochs - ep - 1,
                           max(args.patience - patience, 0))
        eta_s = remaining_ep * avg_ep
        eta_m, eta_sec = int(eta_s // 60), int(eta_s % 60)
        mark = "★" if is_best else " "
        extra = ""
        if vq_stats:
            extra = (f" | vq_ent={vq_stats['proto_entropy']:.2f} "
                     f"τ={vq_stats['tau']:.2f}")
        if variant == "A3_dual":
            lc = loss_parts_sum.get("L_consist", 0.0) / max(n_batches, 1)
            la = loss_parts_sum.get("L_action", 0.0) / max(n_batches, 1)
            extra = f" | L_act={la:.4f} L_con={lc:.4f}"
        elif variant == "A1":
            lr_p = loss_parts_sum.get("L_role", 0.0) / max(n_batches, 1)
            extra = f" | L_role={lr_p:.4f}"
        print(f"E{ep:03d} {mark} loss={tr_loss/max(n_batches,1):.4f} "
              f"val_ex={val_metrics['exact']:5.1f}% test_ex={test_metrics['exact']:5.1f}% "
              f"tok_t={test_metrics['tok_acc']:5.1f}% lr={sched.get_last_lr()[0]:.2e} "
              f"pat={patience} | {dt:.1f}s ETA {eta_m}m{eta_sec:02d}s{extra}")

        if patience >= args.patience:
            print(f"Early stopping at epoch {ep} (patience {args.patience}).")
            break

    # Final summary
    summary = {
        "variant": variant, "seed": seed, "split": split, "args": vars(args),
        "n_params": n_params, "epochs_run": ep + 1,
        "final_train_loss": metrics_log[-1]["train_loss"],
        "final_val_exact": metrics_log[-1]["val"]["exact"],
        "final_test_exact": metrics_log[-1]["test"]["exact"],
        "best_val_exact": best_val_exact,
        "best_test_exact": max(m["test"]["exact"] for m in metrics_log),
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save checkpoint (state_dict + vocabs) — needed for A2_warm init
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "in_w2i": in_w2i,
        "out_w2i": out_w2i,
        "variant": variant,
        "d_model": args.d_model,
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    print(f"\nFinished {variant}/s{seed}: "
          f"best_test={summary['best_test_exact']:.1f}% "
          f"final_test={summary['final_test_exact']:.1f}%")
    return summary


# ══════════════════════════════════════════════════════════════
# Recap (aggregate over seeds)
# ══════════════════════════════════════════════════════════════
def recap(runs_root):
    from glob import glob
    all_runs = []
    for sub in sorted(os.listdir(runs_root)):
        f = os.path.join(runs_root, sub, "summary.json")
        if os.path.exists(f):
            all_runs.append(json.load(open(f)))
    by_var: Dict[str, list] = {}
    for r in all_runs:
        by_var.setdefault(r["variant"], []).append(r)
    print(f"\n{'='*70}")
    print("SCAN addprim_jump compositional — recap")
    print(f"{'='*70}")
    print(f"{'Variant':<10} {'seeds':>6} {'test_mean':>11} {'test_std':>10} "
          f"{'best_mean':>10} {'seeds_runs':>12}")
    print("-" * 68)
    for var in sorted(by_var):
        runs = by_var[var]
        finals = [r["final_test_exact"] for r in runs]
        bests  = [r["best_test_exact"]  for r in runs]
        print(f"{var:<10} {len(runs):>6} {np.mean(finals):>10.1f}% "
              f"{np.std(finals):>9.2f}% {np.mean(bests):>9.1f}% "
              f"{','.join(str(r['seed']) for r in runs):>12}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SCAN addprim_jump compositional ablation")
    p.add_argument("--variant", type=str,
                   choices=["A0", "A1", "A2", "A2_nosup", "A2_warm", "A2_pre",
                            "A3_dual", "A4_permute", "A4_seen", "A4_all"])
    p.add_argument("--split", type=str, default="addprim_jump",
                   choices=["addprim_jump", "addprim_turn_left", "length"],
                   help="SCAN split to use")
    p.add_argument("--a1-checkpoint", type=str, default=None,
                   help="A2_warm only: path to A1 checkpoint.pt")
    p.add_argument("--warm-hold-epochs", type=int, default=50,
                   help="A2_warm only: epochs of τ=2.0 before decay (default 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lambda-role", type=float, default=0.3,
                   help="A1 only: weight of role auxiliary CE loss")
    p.add_argument("--lambda-commit", type=float, default=0.1,
                   help="A2 only: weight of VQ commitment loss")
    p.add_argument("--lambda-ent", type=float, default=0.02,
                   help="A2 only: weight of codebook entropy term")
    p.add_argument("--lambda-consist", type=float, default=1.0,
                   help="A3_dual only: weight of Structure encoder consistency loss")
    p.add_argument("--runs-dir", type=str, default=None,
                   help="Override output directory for runs")
    p.add_argument("--recap", type=str, default=None,
                   help="Path to runs dir to aggregate, e.g. --recap runs/scan_compositional")
    args = p.parse_args()

    if args.recap:
        recap(args.recap)
    elif args.variant:
        train_one_run(args.variant, args.seed, args)
    else:
        p.print_help()
