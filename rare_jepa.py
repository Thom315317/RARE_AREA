#!/usr/bin/env python3
"""
RARE-JEPA: Mixture-of-Experts with Introspective World Model Routing
=====================================================================
A world model that models the network's OWN computation, not the external world.
The JEPA predictor SIMULATES each expert's output in latent space via a learned
expert embedding, scores each simulation with a value head, then routes to the
expert whose simulation predicts the best utility.

Corrections v2:
  - ValueHead replaces max-confidence for routing (predicted utility)
  - Expert embedding (32-dim) replaces one-hot
  - 3-phase curriculum: forced → ε-greedy 0.3 → ε decay 0.3→0.05
  - Coverage forcing for underused experts (<5% → force 10%)

Experiments on bAbI Tasks 1, 2, 3. Single-file standalone.
Hardware target: RTX 3070 8GB VRAM, CUDA + AMP fp16.
"""

import os
import sys
import math
import time
import copy
import json
import random
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    LIVE_PLOT = True
except Exception:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LIVE_PLOT = False

# AMP compatibility (PyTorch >=2.0 vs older)
try:
    _amp_autocast = lambda en: torch.amp.autocast("cuda", enabled=en)
    _amp_scaler = lambda en: torch.amp.GradScaler("cuda", enabled=en)
    _amp_scaler(False)  # probe
except Exception:
    from torch.cuda.amp import autocast as _ac, GradScaler as _GS
    _amp_autocast = lambda en: _ac(enabled=en)
    _amp_scaler = lambda en: _GS(enabled=en)

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
SEED         = 42
HIDDEN_DIM   = 256
N_EXPERTS    = 6
N_HEADS      = 4
FFN_DIM      = 512
MAX_LEN      = 512
MAX_STEPS    = 8
BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PHASE1_END   = 20   # epoch 0-19 : forced sequential
PHASE2_END   = 30   # epoch 20-29: ε-greedy 0.3
TOTAL_EPOCHS = 35   # epoch 30-34: ε decay 0.3→0.05
N_TRAIN      = 3000
N_TEST       = 600
DROPOUT      = 0.2
ENT_COEFF    = 0.05
AUX_COEFF    = 0.2
JEPA_COEFF   = 0.1
VALUE_COEFF  = 0.1
EXPERT_EMB   = 32
MIN_STEPS    = 2     # no halt before step 2
EMA_DECAY    = 0.99  # EMA for JEPA target stability
JEPA_COEFF_COLEARN = 0.05  # reduced weight when co-learning (experts get L_jepa grad)
COVERAGE_THR = 0.10   # force if <10%
COVERAGE_PCT = 0.20   # on 20% of samples

# Barycentric routing
HALT_CONF_THR   = 0.95   # halt when classifier(bary) max softmax > threshold
CCV_COEFF       = 0.01   # L_ccv weight (encourage convergence)
CCV_THR_FIXED   = 0.3    # fallback threshold if --ccv-fixed

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


def _json_clean(obj):
    """Recursively convert numpy/tensor objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def save_run(name: str, data: dict, subfolder: str = None) -> str:
    """Save run data to runs/[subfolder/]<timestamp>_<name>.json (no overwrite)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR
    if subfolder:
        run_dir = os.path.join(RUNS_DIR, subfolder)
        os.makedirs(run_dir, exist_ok=True)
    fn = os.path.join(run_dir, f"{ts}_{name}.json")
    payload = {
        "timestamp": ts,
        "name": name,
        "config": {
            "SEED": SEED, "HIDDEN_DIM": HIDDEN_DIM, "N_EXPERTS": N_EXPERTS,
            "N_HEADS": N_HEADS, "FFN_DIM": FFN_DIM, "MAX_STEPS": MAX_STEPS,
            "BATCH_SIZE": BATCH_SIZE, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "PHASE1_END": PHASE1_END, "PHASE2_END": PHASE2_END,
            "TOTAL_EPOCHS": TOTAL_EPOCHS, "N_TRAIN": N_TRAIN, "N_TEST": N_TEST,
            "DROPOUT": DROPOUT, "ENT_COEFF": ENT_COEFF, "AUX_COEFF": AUX_COEFF,
            "JEPA_COEFF": JEPA_COEFF, "VALUE_COEFF": VALUE_COEFF,
            "EXPERT_EMB": EXPERT_EMB, "MIN_STEPS": MIN_STEPS,
            "EMA_DECAY": EMA_DECAY, "COVERAGE_THR": COVERAGE_THR,
            "COVERAGE_PCT": COVERAGE_PCT,
        },
        "data": _json_clean(data),
    }
    with open(fn, "w") as f:
        json.dump(payload, f, indent=2)
    return fn

# ══════════════════════════════════════════════════════════════
# Seed & Device
# ══════════════════════════════════════════════════════════════

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"Device: {DEVICE} | AMP: {USE_AMP}")

# ══════════════════════════════════════════════════════════════
# bAbI Synthetic Data Generation
# ══════════════════════════════════════════════════════════════
PEOPLE     = ["mary", "john", "sandra", "daniel"]
LOCATIONS  = ["bathroom", "hallway", "kitchen", "garden", "bedroom", "office"]
OBJECTS    = ["football", "milk", "apple"]
MOVE_VERBS = ["moved to", "went to", "journeyed to", "travelled to"]


def _gen_task1(n: int, rng: random.Random) -> List[Tuple[str, str, str]]:
    """Task 1: single supporting fact — where is [person]?
    Anti-shortcut: ALWAYS 1-3 distractor sentences AFTER the target's last mention.
    The 'predict last sentence location' heuristic must fail."""
    out = []
    for _ in range(n):
        sents = []
        target = rng.choice(PEOPLE)
        others = [p for p in PEOPLE if p != target]

        # Before: random sentences (target may appear → earlier locations)
        for _ in range(rng.randint(1, 3)):
            p = rng.choice(PEOPLE)
            sents.append(f"{p} {rng.choice(MOVE_VERBS)} the {rng.choice(LOCATIONS)} .")

        # KEY fact: target's last (answer-defining) location
        answer_loc = rng.choice(LOCATIONS)
        sents.append(f"{target} {rng.choice(MOVE_VERBS)} the {answer_loc} .")

        # AFTER: 1-3 distractors about OTHER people only (never target)
        for _ in range(rng.randint(1, 3)):
            p = rng.choice(others)
            sents.append(f"{p} {rng.choice(MOVE_VERBS)} the {rng.choice(LOCATIONS)} .")

        out.append((" ".join(sents), f"where is {target} ?", answer_loc))
    return out


def _gen_task2(n: int, rng: random.Random) -> List[Tuple[str, str, str]]:
    """Task 2: two supporting facts — object tracking, one move.
    Anti-shortcut: ALWAYS 1-2 distractors after the key move."""
    out = []
    for _ in range(n):
        sents = []
        person = rng.choice(PEOPLE)
        obj = rng.choice(OBJECTS)
        others = [q for q in PEOPLE if q != person]

        sents.append(f"{person} picked up the {obj} .")
        # Distractors before move
        for _ in range(rng.randint(0, 2)):
            sents.append(f"{rng.choice(others)} {rng.choice(MOVE_VERBS)} the {rng.choice(LOCATIONS)} .")
        # KEY: person moves (answer location)
        loc = rng.choice(LOCATIONS)
        sents.append(f"{person} {rng.choice(MOVE_VERBS)} the {loc} .")
        # ALWAYS 1-2 distractors after (other people only)
        for _ in range(rng.randint(1, 2)):
            sents.append(f"{rng.choice(others)} {rng.choice(MOVE_VERBS)} the {rng.choice(LOCATIONS)} .")

        out.append((" ".join(sents), f"where is the {obj} ?", loc))
    return out


def _gen_task3(n: int, rng: random.Random) -> List[Tuple[str, str, str]]:
    """Task 3: three supporting facts — object tracking, 2-3 moves.
    Anti-shortcut: ALWAYS 1-2 distractors after the final move."""
    out = []
    for _ in range(n):
        sents = []
        person = rng.choice(PEOPLE)
        obj = rng.choice(OBJECTS)
        others = [q for q in PEOPLE if q != person]

        sents.append(f"{person} picked up the {obj} .")
        n_moves = rng.randint(2, 3)
        final = None
        for i in range(n_moves):
            loc = rng.choice(LOCATIONS)
            sents.append(f"{person} {rng.choice(MOVE_VERBS)} the {loc} .")
            final = loc
            # Interleave distractors between moves (not after last)
            if i < n_moves - 1 and rng.random() < 0.5:
                sents.append(f"{rng.choice(others)} {rng.choice(MOVE_VERBS)} the {rng.choice(LOCATIONS)} .")
        # ALWAYS 1-2 distractors after the final move
        for _ in range(rng.randint(1, 2)):
            sents.append(f"{rng.choice(others)} {rng.choice(MOVE_VERBS)} the {rng.choice(LOCATIONS)} .")

        out.append((" ".join(sents), f"where is the {obj} ?", final))
    return out


TASK_GEN = {1: _gen_task1, 2: _gen_task2, 3: _gen_task3}

# ══════════════════════════════════════════════════════════════
# Vocabulary & Dataset
# ══════════════════════════════════════════════════════════════

class Vocab:
    def __init__(self):
        self.w2i: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.i2w: Dict[int, str] = {0: "<pad>", 1: "<unk>"}

    def add(self, w: str):
        if w not in self.w2i:
            i = len(self.w2i)
            self.w2i[w] = i
            self.i2w[i] = w

    def encode(self, text: str) -> List[int]:
        return [self.w2i.get(w, 1) for w in text.lower().split()]

    def __len__(self):
        return len(self.w2i)


def build_vocab(all_samples_lists):
    v = Vocab()
    for samples in all_samples_lists:
        for s, q, a in samples:
            for w in (s + " " + q + " " + a).lower().split():
                v.add(w)
    return v


class BabiDataset(Dataset):
    def __init__(self, samples, vocab: Vocab):
        self.data = []
        for s, q, a in samples:
            toks = vocab.encode(s + " " + q)[:MAX_LEN]
            tgt = vocab.w2i.get(a.lower(), 1)
            self.data.append((toks, tgt, len(toks)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    toks, tgts, lens = zip(*batch)
    ml = max(lens)
    padded = torch.zeros(len(batch), ml, dtype=torch.long)
    mask   = torch.zeros(len(batch), ml, dtype=torch.bool)
    for i, (t, _, l) in enumerate(zip(toks, tgts, lens)):
        padded[i, :l] = torch.tensor(t, dtype=torch.long)
        mask[i, :l] = True
    return padded, mask, torch.tensor(tgts, dtype=torch.long), torch.tensor(lens, dtype=torch.long)

# ══════════════════════════════════════════════════════════════
# Model Components
# ══════════════════════════════════════════════════════════════

class TransformerExpert(nn.Module):
    """Pre-LayerNorm Transformer block (one expert)."""
    def __init__(self, dim=None, heads=None, ffn=None, drop=None):
        super().__init__()
        if dim is None: dim = HIDDEN_DIM
        if heads is None: heads = N_HEADS
        if ffn is None: ffn = FFN_DIM
        if drop is None: drop = DROPOUT
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(),
                                   nn.Linear(ffn, dim), nn.Dropout(drop))

    def forward(self, x, mask=None):
        kpm = ~mask if mask is not None else None
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=kpm)
        x = x + self.drop1(h)
        x = x + self.ffn(self.norm2(x))
        return x


class JEPAPredictor(nn.Module):
    """Introspective world model — predicts what each expert WOULD produce."""
    def __init__(self, dim=None, edim=None):
        super().__init__()
        if dim is None: dim = HIDDEN_DIM
        if edim is None: edim = EXPERT_EMB
        self.net = nn.Sequential(
            nn.Linear(dim + dim + edim, dim),  # z + m + expert_emb = 544
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z, m, e_emb):
        # Harmonize dtypes (AMP may produce fp16 z but fp32 m)
        dt = z.dtype
        return self.net(torch.cat([z, m.to(dt), e_emb.to(dt)], dim=-1))


class ValueHead(nn.Module):
    """Predicts utility of a latent state for classification."""
    def __init__(self, dim=None, inner=64):
        super().__init__()
        if dim is None: dim = HIDDEN_DIM
        self.net = nn.Sequential(nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)  # (B,)


class GumbelRouter(nn.Module):
    """Standard MLP router for Gumbel-ST baseline (Exp1)."""
    def __init__(self, dim=None, n_opt=N_EXPERTS + 1):
        super().__init__()
        if dim is None: dim = HIDDEN_DIM
        self.net = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, n_opt))

    def forward(self, z, m):
        return self.net(torch.cat([z, m.to(z.dtype)], dim=-1))


class UnifiedRouter(nn.Module):
    """Router for RARE-UNIFIED (dedup): 6 actions = go to expert k.
    advance/revise/jump classified post-hoc based on chosen vs current.
    Takes rich state: z, m, current_expert, step_idx, n_visits."""
    STEP_DIM = 16

    def __init__(self, dim=None, n_experts=N_EXPERTS, max_steps=MAX_STEPS):
        super().__init__()
        if dim is None: dim = HIDDEN_DIM
        self.n_experts = n_experts
        self.n_actions = n_experts              # one output per expert
        self.step_emb = nn.Embedding(max_steps + 1, self.STEP_DIM)
        in_dim = dim + dim + n_experts + self.STEP_DIM + n_experts
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.n_actions),
        )

    def forward(self, z, m, current, step, n_visits):
        # z, m: (B, D)  current: (B,) long  step: int  n_visits: (B, K)
        B = z.size(0)
        dt = z.dtype
        cur_oh  = F.one_hot(current, self.n_experts).to(dt)
        step_t  = torch.tensor(step, device=z.device, dtype=torch.long)
        step_e  = self.step_emb(step_t).to(dt).unsqueeze(0).expand(B, -1)
        n_vis_n = (n_visits.to(dt) / float(step + 1))
        state   = torch.cat([z, m.to(dt), cur_oh, step_e, n_vis_n], dim=-1)
        return self.net(state)                   # (B, n_actions)

# ══════════════════════════════════════════════════════════════
# RARE-JEPA  —  Main Model
# ══════════════════════════════════════════════════════════════

class RAREJEPA(nn.Module):
    """
    routing: 'sequential' | 'gumbel' | 'jepa'
    use_gru: False → m always zeros (Exp3 ablation)
    """
    def __init__(self, vocab_size: int, routing: str = "jepa",
                 use_gru: bool = True, allow_revise: bool = True,
                 hybrid: bool = False, ccv_learned: bool = True,
                 jepa_colearn: bool = False):
        super().__init__()
        D = HIDDEN_DIM
        self.routing      = routing
        self.use_gru      = use_gru
        self.allow_revise = allow_revise
        self.hybrid       = hybrid          # Gumbel-softmax on JEPA value_scores
        self.ccv_learned  = ccv_learned     # learnable vs fixed CCV threshold
        self.jepa_colearn = jepa_colearn    # let L_jepa gradient flow to experts
        self.D            = D
        self.n_exp        = N_EXPERTS
        self.max_t        = MAX_STEPS

        # Embedding
        self.tok_emb  = nn.Embedding(vocab_size, D, padding_idx=0)
        self.pos_emb  = nn.Embedding(MAX_LEN, D)
        self.emb_norm = nn.LayerNorm(D)
        self.emb_drop = nn.Dropout(DROPOUT)

        # Experts (6 separate Transformer blocks)
        self.experts = nn.ModuleList([TransformerExpert() for _ in range(N_EXPERTS)])

        # GRU memory
        self.gru = nn.GRUCell(D, D)

        # Classifier (shared: final + deep supervision + value reward)
        self.classifier = nn.Linear(D, vocab_size)

        # -- JEPA-specific --
        if routing in ("jepa", "bary", "meta", "unified"):
            self.expert_emb = nn.Embedding(N_EXPERTS, EXPERT_EMB)
            self.jepa       = JEPAPredictor()
            # EMA shadow of experts for stable JEPA targets
            self.ema_experts = copy.deepcopy(self.experts)
            for p in self.ema_experts.parameters():
                p.requires_grad_(False)

        if routing in ("jepa", "meta", "unified"):
            self.value_head = ValueHead()

        if routing in ("bary", "meta", "unified"):
            # Learnable CCV threshold (init to sigmoid^-1(0.3) ≈ -0.85)
            init = math.log(CCV_THR_FIXED / (1 - CCV_THR_FIXED))
            self.ccv_threshold_param = nn.Parameter(torch.tensor(init))

        # -- Gumbel-specific --
        if routing in ("gumbel", "meta", "unified"):
            self.gumbel_router = GumbelRouter()

        # -- Unified-specific --
        if routing == "unified":
            self.unified_router = UnifiedRouter()
            self.unified_submode = "jepa"     # default, changed by pilot
            self.action_mask_mode = "all"     # "all" | "adv_rev" | "jump"

        # -- Meta mode: can switch routing at training time --
        self.is_meta = (routing == "meta")
        if self.is_meta:
            # Default to "jepa" for Phase 1 (to train JEPA predictor)
            self.routing = "jepa"

    # ── helpers ───────────────────────────────────────────────
    def _pool(self, x, mask):
        """Attention pooling: last real token attends to full sequence."""
        # Find last real token index per sample
        # mask: (B, L) bool
        lengths = mask.sum(1)                      # (B,)
        last_idx = (lengths - 1).clamp(min=0)      # (B,)
        # Extract last token repr as query
        q = x[torch.arange(x.size(0), device=x.device), last_idx]  # (B, D)
        # Attention scores: q dot each token
        scores = torch.bmm(q.unsqueeze(1), x.transpose(1, 2)).squeeze(1)  # (B, L)
        scores = scores.masked_fill(~mask, -6e4)  # fp16-safe (max ~65504)
        attn = F.softmax(scores, dim=-1)           # (B, L)
        return torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # (B, D)

    # ── forward ───────────────────────────────────────────────
    def forward(self, tokens, mask, phase=1, epsilon=0.0,
                underused_experts=None, targets=None):
        """
        Returns (logits, info).
        Pass targets for value-head training signal.
        """
        B, L = tokens.shape
        dev  = tokens.device
        D    = self.D
        K    = self.n_exp

        # Embed
        pos = torch.arange(L, device=dev).unsqueeze(0)
        x   = self.emb_drop(self.emb_norm(self.tok_emb(tokens) + self.pos_emb(pos)))

        # State
        m       = torch.zeros(B, D, device=dev)
        halted  = torch.zeros(B, dtype=torch.bool, device=dev)
        h_steps = torch.full((B,), self.max_t, dtype=torch.long, device=dev)

        info = {
            "jepa_losses": [], "value_losses": [], "ccv_losses": [],
            "aux_logits_active": [],
            "expert_choices": [],
            "expert_active":  [],
            "halt_steps": h_steps,
            "jepa_errs_per_expert": defaultdict(list),
            "jepa_errs_per_step":   defaultdict(list),
            "ccv_per_step":         defaultdict(list),   # barycentric
            "stability_per_step":   defaultdict(list),   # % samples in stability mode
            "dist_to_bary_per_step": defaultdict(list),  # chosen expert's dist
            "_jepa_coeff": (JEPA_COEFF_COLEARN if self.jepa_colearn else JEPA_COEFF),
        }

        # No-revise: track which experts have been used per sample
        used_mask = torch.zeros(B, K, dtype=torch.bool, device=dev)

        # Unified: track current_expert + visit count per expert
        current_expert = torch.zeros(B, dtype=torch.long, device=dev)
        n_visits       = torch.zeros(B, K, dtype=torch.long, device=dev)
        info["action_counts"] = defaultdict(int)  # adv, rev, jump
        info["action_per_step"] = defaultdict(list)

        # Max steps: K if sequential/phase1, K if no-revise, else MAX_STEPS
        if self.routing == "sequential" or phase == 1:
            n_iter = K
        elif not self.allow_revise:
            n_iter = K          # can't use more than K if each expert once
        else:
            n_iter = self.max_t

        # Phase 1: shuffled order per batch (break co-adaptation)
        if phase == 1:
            p1_perm = torch.randperm(K, device=dev)

        for t in range(n_iter):
            prev_halted = halted.clone()
            deciding    = ~prev_halted
            if not deciding.any():
                break

            z    = self._pool(x, mask)          # (B, D)
            m_in = m if self.use_gru else torch.zeros_like(m)
            jepa_pred = None
            pred_score = None

            # ── Routing ───────────────────────────────────────
            if phase == 1 or self.routing == "sequential":
                # Shuffled sequential: random permutation per batch
                if self.routing == "sequential":
                    chosen = torch.full((B,), t % K, dtype=torch.long, device=dev)
                else:
                    chosen = torch.full((B,), p1_perm[t % K].item(),
                                        dtype=torch.long, device=dev)
                is_halt = torch.zeros(B, dtype=torch.bool, device=dev)

                # JEPA observation during Phase 1
                # (also for bary/unified, to train their shared JEPA predictor)
                if self.routing in ("jepa", "bary", "unified"):
                    e = self.expert_emb(chosen)
                    jepa_pred = self.jepa(z, m_in, e)
                    # Also train value head for jepa and unified (both use it)
                    if self.routing in ("jepa", "unified"):
                        pred_score = self.value_head(jepa_pred)

            elif self.routing == "gumbel":
                logits_all = self.gumbel_router(z, m_in)    # (B, K+1)
                if t < MIN_STEPS:
                    logits_all = logits_all.clone()
                    logits_all[:, K] = -1e9

                if self.training:
                    probs = F.gumbel_softmax(logits_all, tau=1.0, hard=True)
                else:
                    probs = F.one_hot(logits_all.argmax(-1), K + 1).float()

                is_halt = probs[:, K] > 0.5
                chosen = probs[:, :K].argmax(-1)

                # Coverage forcing
                if (self.training and underused_experts is not None
                        and len(underused_experts) > 0):
                    cm = torch.rand(B, device=dev) < COVERAGE_PCT
                    fe = underused_experts[torch.randint(
                        0, len(underused_experts), (B,), device=dev)]
                    chosen = torch.where(cm & deciding, fe, chosen)

            elif self.routing == "jepa":
                # ── Batched JEPA simulation ──
                z_rep = z.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
                m_rep = m_in.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
                eidx  = torch.arange(K, device=dev).unsqueeze(0).expand(B, K).reshape(B * K)
                e_all = self.expert_emb(eidx)
                all_preds = self.jepa(z_rep, m_rep, e_all).view(B, K, D)
                expert_scores = self.value_head(
                    all_preds.reshape(B * K, D)).view(B, K)
                halt_score = self.value_head(z)
                all_scores = torch.cat([expert_scores,
                                        halt_score.unsqueeze(1)], 1)

                if t < MIN_STEPS:
                    all_scores = all_scores.clone()
                    all_scores[:, K] = -1e9

                # No-revise: mask already-used experts
                if not self.allow_revise:
                    all_scores = all_scores.clone()
                    all_scores[:, :K][used_mask] = -1e9
                    # If all experts used → force halt
                    all_used = used_mask.all(dim=1)
                    all_scores[all_used, K] = 1e9

                # HYBRID: Gumbel-Softmax on JEPA value scores
                # Higher temperature at step 0 (JEPA unreliable), lower later
                if self.hybrid and self.training:
                    tau = 2.0 if t == 0 else (1.0 if t == 1 else 0.5)
                    probs_hyb = F.gumbel_softmax(all_scores / tau, hard=True)
                    chosen_full = probs_hyb.argmax(1)
                else:
                    chosen_full = all_scores.argmax(1)

                # ε-greedy (only among available experts)
                if self.training and epsilon > 0:
                    rm = torch.rand(B, device=dev) < epsilon
                    if not self.allow_revise:
                        # Random among unused experts
                        avail = (~used_mask).float()
                        avail_norm = avail / avail.sum(1, keepdim=True).clamp(min=1)
                        rand_exp = torch.multinomial(avail_norm, 1).squeeze(1)
                    else:
                        rand_exp = torch.randint(0, K, (B,), device=dev)
                    chosen_full = torch.where(rm & deciding, rand_exp, chosen_full)

                # Coverage forcing
                if (self.training and underused_experts is not None
                        and len(underused_experts) > 0):
                    cm = torch.rand(B, device=dev) < COVERAGE_PCT
                    fe = underused_experts[torch.randint(
                        0, len(underused_experts), (B,), device=dev)]
                    chosen_full = torch.where(cm & deciding, fe, chosen_full)

                is_halt = chosen_full == K
                chosen  = chosen_full.clamp(max=K - 1)
                idx     = chosen.view(B, 1, 1).expand(-1, 1, D)
                jepa_pred  = all_preds.gather(1, idx).squeeze(1)
                pred_score = all_scores.gather(
                    1, chosen_full.unsqueeze(1)).squeeze(1)

            elif self.routing == "bary":
                # ── Batched JEPA predictions ──────────────────────
                z_rep = z.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
                m_rep = m_in.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
                eidx  = torch.arange(K, device=dev).unsqueeze(0).expand(B, K).reshape(B * K)
                e_all = self.expert_emb(eidx)
                all_preds = self.jepa(z_rep, m_rep, e_all).view(B, K, D)  # (B, K, D)

                # Barycenter of predictions
                bary = all_preds.mean(dim=1)                         # (B, D)

                # Normalize for cosine distances
                preds_n = F.normalize(all_preds, dim=-1)              # (B, K, D)
                bary_n  = F.normalize(bary, dim=-1).unsqueeze(1)      # (B, 1, D)

                # Distance of each prediction to barycenter
                dist_to_bary = 1.0 - (preds_n * bary_n).sum(-1)       # (B, K)

                # CCV: mean pairwise cosine distance between predictions
                sim = torch.bmm(preds_n, preds_n.transpose(1, 2))     # (B, K, K)
                eye = torch.eye(K, device=dev).unsqueeze(0)
                ccv = ((1.0 - sim) * (1.0 - eye)).sum(dim=(1, 2)) / (K * (K - 1))  # (B,)

                # Threshold: learned sigmoid or fixed
                if self.ccv_learned:
                    thr = torch.sigmoid(self.ccv_threshold_param)
                else:
                    thr = torch.tensor(CCV_THR_FIXED, device=dev)

                # Mode: stability if CCV < threshold, else exploration
                stability = ccv < thr                                 # (B,) bool

                # Score: negative dist (central) in stability, positive (divergent) in expl
                score = torch.where(stability.unsqueeze(1),
                                    -dist_to_bary, dist_to_bary)

                chosen = score.argmax(-1)

                # ε-greedy
                if self.training and epsilon > 0:
                    rm = torch.rand(B, device=dev) < epsilon
                    rand_exp = torch.randint(0, K, (B,), device=dev)
                    chosen = torch.where(rm & deciding, rand_exp, chosen)

                # Halt by classifier confidence on barycenter
                halt_logits = self.classifier(bary)
                halt_conf   = F.softmax(halt_logits, dim=-1).max(-1).values  # (B,)
                is_halt = halt_conf > HALT_CONF_THR
                if t < MIN_STEPS:
                    is_halt = torch.zeros_like(is_halt)

                # Chosen JEPA pred (for L_jepa)
                idx = chosen.view(B, 1, 1).expand(-1, 1, D)
                jepa_pred = all_preds.gather(1, idx).squeeze(1)

                # Track CCV loss (encourage convergence globally)
                info["ccv_losses"].append(ccv.mean())

                # Metrics
                info["ccv_per_step"][t].append(ccv.mean().item())
                info["stability_per_step"][t].append(stability.float().mean().item())
                chosen_dist = dist_to_bary.gather(1, chosen.unsqueeze(1)).squeeze(1)
                info["dist_to_bary_per_step"][t].append(chosen_dist.mean().item())

            elif self.routing == "unified":
                # ── Unified routing (DEDUP): 6 actions = choose an expert ──
                # advance/revise/jump classified post-hoc
                router_logits = self.unified_router(
                    z, m_in, current_expert, t, n_visits)        # (B, K)

                # Action masking (restrict allowed experts per mode)
                mask_logits = router_logits.clone()
                if self.action_mask_mode == "adv_rev":
                    # Only allow current (revise) and (current+1)%K (advance)
                    allowed = torch.zeros(B, K, dtype=torch.bool, device=dev)
                    allowed.scatter_(1, current_expert.unsqueeze(1), True)
                    allowed.scatter_(1, ((current_expert + 1) % K).unsqueeze(1), True)
                    mask_logits = torch.where(
                        allowed, mask_logits,
                        torch.full_like(mask_logits, -6e4))
                # "all" and "jump" → no masking (all 6 experts allowed)

                # JEPA predictions for the 6 experts (needed by jepa/bary submodes)
                z_rep = z.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
                m_rep = m_in.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
                eidx  = torch.arange(K, device=dev).unsqueeze(0).expand(B, K).reshape(B * K)
                e_all = self.expert_emb(eidx)
                all_preds = self.jepa(z_rep, m_rep, e_all).view(B, K, D)

                # ── Submode ──────────────────────────────────────
                sm = self.unified_submode
                if sm == "gumbel":
                    if self.training:
                        probs = F.gumbel_softmax(mask_logits, tau=1.0, hard=True)
                    else:
                        probs = F.one_hot(mask_logits.argmax(-1), K).float()
                    chosen = probs.argmax(-1)

                elif sm == "jepa":
                    # ValueHead scores on each expert's prediction
                    v_scores = self.value_head(
                        all_preds.reshape(B * K, D)).view(B, K)
                    v_scores = torch.where(
                        mask_logits < -1e3,
                        torch.full_like(v_scores, -6e4),
                        v_scores)
                    chosen = v_scores.argmax(-1)

                elif sm == "bary":
                    bary = all_preds.mean(dim=1)
                    preds_n = F.normalize(all_preds, dim=-1)
                    bary_n  = F.normalize(bary, dim=-1).unsqueeze(1)
                    dist_to_bary = 1.0 - (preds_n * bary_n).sum(-1)       # (B, K)
                    sim = torch.bmm(preds_n, preds_n.transpose(1, 2))
                    eye = torch.eye(K, device=dev).unsqueeze(0)
                    ccv = ((1.0 - sim) * (1.0 - eye)).sum(dim=(1, 2)) / (K * (K - 1))
                    thr = torch.sigmoid(self.ccv_threshold_param)
                    stability = ccv < thr
                    # Expert score: -dist if stable, +dist if explore
                    expert_score = torch.where(
                        stability.unsqueeze(1),
                        -dist_to_bary, dist_to_bary)
                    expert_score = torch.where(
                        mask_logits < -1e3,
                        torch.full_like(expert_score, -6e4),
                        expert_score)
                    chosen = expert_score.argmax(-1)
                    info["ccv_losses"].append(ccv.mean())
                    info["ccv_per_step"][t].append(ccv.mean().item())

                else:
                    raise ValueError(f"Unknown unified submode: {sm}")

                # ε-greedy among allowed experts
                if self.training and epsilon > 0:
                    rm = torch.rand(B, device=dev) < epsilon
                    if self.action_mask_mode == "adv_rev":
                        # Random in {current, (current+1) % K}
                        which = torch.randint(0, 2, (B,), device=dev)
                        rand_exp = torch.where(
                            which == 0, current_expert,
                            (current_expert + 1) % K)
                    else:
                        rand_exp = torch.randint(0, K, (B,), device=dev)
                    chosen = torch.where(rm & deciding, rand_exp, chosen)

                is_halt = torch.zeros(B, dtype=torch.bool, device=dev)

                # JEPA pred for chosen expert (for L_jepa)
                idx = chosen.view(B, 1, 1).expand(-1, 1, D)
                jepa_pred = all_preds.gather(1, idx).squeeze(1)

                # Post-hoc action classification (for metrics)
                is_adv = (chosen == (current_expert + 1) % K)
                is_rev = (chosen == current_expert)
                info["action_counts"]["advance"] += is_adv.sum().item()
                info["action_counts"]["revise"]  += is_rev.sum().item()
                info["action_counts"]["jump"]    += (~is_adv & ~is_rev).sum().item()
                info["action_per_step"][t].append(chosen.cpu().tolist())

            # Update used_mask (no in-place to avoid autograd conflict)
            used_mask = used_mask | F.one_hot(chosen, K).bool()
            # Update unified state
            n_visits = n_visits + F.one_hot(chosen, K).long()
            current_expert = chosen

            # ── Halt bookkeeping ──────────────────────────────
            newly_halted = is_halt & deciding
            h_steps[newly_halted] = t
            halted = halted | is_halt
            active = deciding & ~is_halt

            if not active.any() and not newly_halted.any():
                info["expert_choices"].append(chosen.clone())
                info["expert_active"].append(active.clone())
                break

            # ── Expert forward (hard routing, one expert per sample) ──
            z_real = x.clone()
            for k in range(K):
                km = (chosen == k) & active
                if km.any():
                    z_real[km] = self.experts[k](x[km], mask[km])

            h_real = self._pool(z_real, mask)

            # ── JEPA loss (predict vs EMA target for stability) ──
            # Compute EMA target every step in Phase 1, every other step in Phase 2+
            compute_jepa = (jepa_pred is not None and active.any()
                            and (phase == 1 or t % 2 == 0))
            if compute_jepa:
                if self.jepa_colearn:
                    # Co-learning: gradient flows back to experts via h_real
                    target_all = h_real
                else:
                    # Standard: EMA target, no gradient to experts
                    with torch.no_grad():
                        z_ema = x.clone()
                        for k in range(K):
                            km = (chosen == k) & active
                            if km.any():
                                z_ema[km] = self.ema_experts[k](x[km], mask[km])
                        target_all = self._pool(z_ema, mask)

                je = F.mse_loss(jepa_pred[active], target_all[active])
                info["jepa_losses"].append(je)
                # Per-step error for diagnostic (H1: error accumulation)
                info["jepa_errs_per_step"][t].append(je.item())
                for k in range(K):
                    ka = (chosen == k) & active
                    if ka.any():
                        # .item() removes grad anyway; use .detach() for clarity
                        info["jepa_errs_per_expert"][k].append(
                            F.mse_loss(jepa_pred[ka].detach(),
                                       target_all[ka].detach()).item())

            # ── Value loss ────────────────────────────────────
            if (self.routing in ("jepa", "unified") and targets is not None
                    and deciding.any() and pred_score is not None):
                with torch.no_grad():
                    rew = -F.cross_entropy(
                        self.classifier(h_real[deciding]),
                        targets[deciding], reduction="none")
                info["value_losses"].append(
                    F.mse_loss(pred_score[deciding], rew))

            # ── Deep supervision ──────────────────────────────
            if active.any():
                info["aux_logits_active"].append(
                    (self.classifier(h_real[active]), active))

            # ── Memory update (fp32 for stability) ────────────
            if self.use_gru and active.any():
                m_new = m.clone()
                m_new[active] = self.gru(
                    h_real[active].float(), m[active]).float()
                m = m_new

            # ── Propagate ─────────────────────────────────────
            x_new = x.clone()
            x_new[active] = z_real[active]
            x = x_new

            info["expert_choices"].append(chosen.clone())
            info["expert_active"].append(active.clone())

        if self.routing == "sequential" or phase == 1:
            info["halt_steps"].fill_(n_iter)

        pooled = self._pool(x, mask)
        info["pooled"] = pooled                # exposed for downstream use (SCAN etc.)
        info["final_seq"] = x                  # full token-level encoded sequence
        logits = self.classifier(pooled)
        return logits, info

# ══════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════

def compute_loss(logits, targets, info, device):
    # L_cls — cross-entropy at halt / final step
    L_cls = F.cross_entropy(logits, targets)

    # L_aux — deep supervision at every active step
    L_aux = torch.tensor(0.0, device=device)
    n_aux = 0
    for sub_logits, active_mask in info["aux_logits_active"]:
        L_aux = L_aux + F.cross_entropy(sub_logits, targets[active_mask])
        n_aux += 1
    if n_aux > 0:
        L_aux = L_aux / n_aux

    # L_jepa — JEPA prediction error
    if info["jepa_losses"]:
        L_jepa = sum(info["jepa_losses"]) / len(info["jepa_losses"])
    else:
        L_jepa = torch.tensor(0.0, device=device)

    # L_value — value head regression
    if info["value_losses"]:
        L_value = sum(info["value_losses"]) / len(info["value_losses"])
    else:
        L_value = torch.tensor(0.0, device=device)

    # L_ent — entropy of expert usage distribution (encourage diversity)
    L_ent = torch.tensor(0.0, device=device)
    if info["expert_choices"]:
        all_c  = torch.cat(info["expert_choices"])
        counts = torch.zeros(N_EXPERTS, device=device)
        for k in range(N_EXPERTS):
            counts[k] = (all_c == k).float().sum()
        total = counts.sum().clamp(min=1)
        p = counts / total
        entropy = -(p * (p + 1e-10).log()).sum()
        L_ent = -entropy

    # L_ccv — barycentric convergence (CRISTAL-inspired)
    L_ccv = torch.tensor(0.0, device=device)
    if info.get("ccv_losses"):
        L_ccv = sum(info["ccv_losses"]) / len(info["ccv_losses"])

    # JEPA coeff override (for co-learning): reduces weight to avoid expert collapse
    jepa_c = info.get("_jepa_coeff", JEPA_COEFF)

    total = (L_cls
             + AUX_COEFF   * L_aux
             + jepa_c      * L_jepa
             + VALUE_COEFF * L_value
             + ENT_COEFF   * L_ent
             + CCV_COEFF   * L_ccv)
    parts = {
        "L_cls": L_cls.item(), "L_aux": L_aux.item(),
        "L_jepa": L_jepa.item(), "L_value": L_value.item(),
        "L_ent": L_ent.item(), "L_ccv": L_ccv.item(),
        "jepa_coeff": jepa_c,
    }
    return total, parts

# ══════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════

def get_phase_eps(routing: str, epoch: int):
    """Return (phase, epsilon) for the current epoch."""
    if routing == "sequential":
        return 1, 0.0
    if epoch < PHASE1_END:
        return 1, 0.0
    if routing == "gumbel":
        return 2, 0.0
    # JEPA routing
    if epoch < PHASE2_END:
        return 2, 0.3
    # Phase 3: linear decay 0.3 → 0.05
    progress = (epoch - PHASE2_END) / max(TOTAL_EPOCHS - PHASE2_END - 1, 1)
    eps = 0.3 - progress * (0.3 - 0.05)
    return 3, eps


def train_epoch(model, loader, optimizer, scaler, phase, epsilon,
                underused):
    model.train()
    tot_loss, tot_ok, tot_n = 0.0, 0, 0
    loss_acc = defaultdict(float)
    ep_choices = []    # flat list of chosen expert ids (int)
    ep_halts   = []

    ue_tensor = (torch.tensor(underused, dtype=torch.long, device=DEVICE)
                 if underused else None)

    for tokens, mask, targets, lengths in loader:
        tokens = tokens.to(DEVICE)
        mask   = mask.to(DEVICE)
        targets = targets.to(DEVICE)

        with _amp_autocast(USE_AMP):
            logits, info = model(tokens, mask, phase=phase, epsilon=epsilon,
                                 underused_experts=ue_tensor, targets=targets)
            loss, parts = compute_loss(logits, targets, info, DEVICE)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # EMA update for JEPA target stability
        if hasattr(model, "ema_experts"):
            with torch.no_grad():
                for p, pe in zip(model.experts.parameters(),
                                 model.ema_experts.parameters()):
                    pe.data.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)

        bs = targets.size(0)
        tot_loss += loss.item() * bs
        tot_ok   += (logits.argmax(-1) == targets).sum().item()
        tot_n    += bs
        for k, v in parts.items():
            loss_acc[k] += v * bs

        # Collect expert choices (only active ones)
        for ch, act in zip(info["expert_choices"], info["expert_active"]):
            ep_choices.extend(ch[act].cpu().tolist())
        ep_halts.extend(info["halt_steps"].cpu().tolist())

    # Aggregate metrics
    m = {"loss": tot_loss / tot_n, "acc": tot_ok / tot_n * 100}
    for k, v in loss_acc.items():
        m[k] = v / tot_n

    # Expert distribution
    ctr = Counter(ep_choices)
    total_c = max(sum(ctr.values()), 1)
    m["expert_dist"] = np.array([ctr.get(k, 0) / total_c for k in range(N_EXPERTS)])
    m["avg_halt"]    = np.mean(ep_halts) if ep_halts else N_EXPERTS

    # Identify underused experts for next epoch
    m["underused"] = [k for k in range(N_EXPERTS)
                      if ctr.get(k, 0) / total_c < COVERAGE_THR]
    return m


@torch.no_grad()
def evaluate(model, loader, phase):
    model.eval()
    tot_ok, tot_n = 0, 0
    # Per-batch storage: list of (choices_list, active_list) per batch
    batch_data = []   # (choices=[step tensors], active=[step tensors], halts, lens)
    jerr = defaultdict(list)
    jerr_per_step = defaultdict(list)  # H1 diagnostic: error vs step t

    for tokens, mask, targets, lengths in loader:
        tokens  = tokens.to(DEVICE)
        mask    = mask.to(DEVICE)
        targets = targets.to(DEVICE)

        with _amp_autocast(USE_AMP):
            logits, info = model(tokens, mask, phase=phase, epsilon=0.0,
                                 targets=targets)

        tot_ok += (logits.argmax(-1) == targets).sum().item()
        tot_n  += targets.size(0)

        batch_data.append((
            [c.cpu() for c in info["expert_choices"]],
            [a.cpu() for a in info["expert_active"]],
            info["halt_steps"].cpu(),
            lengths,
        ))
        for k, v in info["jepa_errs_per_expert"].items():
            jerr[k].extend(v)
        for t_step, v in info["jepa_errs_per_step"].items():
            jerr_per_step[t_step].extend(v)

    m = {"acc": tot_ok / tot_n * 100}

    # ── Expert distribution (flat over all active choices) ────
    flat_ch = []
    for choices, actives, _, _ in batch_data:
        for ch, act in zip(choices, actives):
            flat_ch.extend(ch[act].tolist())
    ctr = Counter(flat_ch)
    tc  = max(sum(ctr.values()), 1)
    m["expert_dist"] = np.array([ctr.get(k, 0) / tc for k in range(N_EXPERTS)])

    # ── Heatmap: expert × step ────────────────────────────────
    heatmap = np.zeros((N_EXPERTS, MAX_STEPS))
    for choices, actives, _, _ in batch_data:
        for step, (ch, act) in enumerate(zip(choices, actives)):
            if step >= MAX_STEPS:
                break
            for k in range(N_EXPERTS):
                heatmap[k, step] += ((ch == k) & act).sum().item()
    m["heatmap"] = heatmap

    # ── Halt steps ────────────────────────────────────────────
    all_halts = torch.cat([bd[2] for bd in batch_data]).float()
    all_lens  = torch.cat([bd[3] for bd in batch_data]).float()
    m["avg_halt"] = all_halts.mean().item()

    # ── Difficulty bins (length-based percentiles) ────────────
    lens_np = all_lens.numpy()
    if len(lens_np) > 2:
        p33, p66 = np.percentile(lens_np, [33, 66])
    else:
        p33 = p66 = float(np.median(lens_np))
    easy = all_lens <= p33
    med  = (all_lens > p33) & (all_lens <= p66)
    hard = all_lens > p66

    # ── Per-sample revise & difficulty tracking ───────────────
    # Rebuild per-sample choice sequences from batch data
    total_revise, total_trans = 0, 0
    rev_by_diff = {"easy": [0, 0], "med": [0, 0], "hard": [0, 0]}
    sample_offset = 0
    for choices, actives, halts_b, lens_b in batch_data:
        bs = halts_b.size(0)
        if len(choices) < 2:
            sample_offset += bs
            continue
        # Stack step-wise choices: (n_steps, bs)
        ch_stack  = torch.stack(choices, 0)
        act_stack = torch.stack(actives, 0)
        for i in range(bs):
            # Get active choices for sample i
            steps_i = [ch_stack[s, i].item() for s in range(ch_stack.size(0))
                       if act_stack[s, i].item()]
            n_rev = sum(1 for j in range(1, len(steps_i))
                        if steps_i[j] == steps_i[j - 1])
            n_tr  = max(len(steps_i) - 1, 0)
            total_revise += n_rev
            total_trans  += n_tr
            # Difficulty bin
            gl_idx = sample_offset + i
            if gl_idx < len(easy):
                if easy[gl_idx]:
                    d = "easy"
                elif hard[gl_idx]:
                    d = "hard"
                else:
                    d = "med"
                rev_by_diff[d][0] += n_rev
                rev_by_diff[d][1] += n_tr
        sample_offset += bs

    m["pct_revise"]  = total_revise / max(total_trans, 1) * 100
    for d in ("easy", "med", "hard"):
        m[f"revise_{d}"] = rev_by_diff[d][0] / max(rev_by_diff[d][1], 1) * 100

    m["halt_easy"] = all_halts[easy].mean().item() if easy.any() else 0
    m["halt_med"]  = all_halts[med].mean().item()  if med.any()  else 0
    m["halt_hard"] = all_halts[hard].mean().item()  if hard.any() else 0

    m["jepa_errs_per_expert"] = {k: np.mean(v) if v else 0.0
                                  for k, v in jerr.items()}
    m["jepa_error_mean"] = (np.mean([np.mean(v) for v in jerr.values()])
                            if jerr else 0.0)
    # H1 diagnostic: per-step error curve
    m["jepa_errs_per_step"] = {t: np.mean(v) if v else 0.0
                                for t, v in jerr_per_step.items()}

    return m

# ══════════════════════════════════════════════════════════════
# Experiment runner
# ══════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "baseline":        {"routing": "sequential", "use_gru": True,  "allow_revise": True,  "hybrid": False},
    "gumbel":          {"routing": "gumbel",     "use_gru": True,  "allow_revise": True,  "hybrid": False},
    "jepa":            {"routing": "jepa",       "use_gru": True,  "allow_revise": True,  "hybrid": False},
    "jepa_hybrid":     {"routing": "jepa",       "use_gru": True,  "allow_revise": True,  "hybrid": True},
    "jepa_norevise":   {"routing": "jepa",       "use_gru": True,  "allow_revise": False, "hybrid": False},
    "jepa_nogru":      {"routing": "jepa",       "use_gru": False, "allow_revise": True,  "hybrid": False},
    "jepa_bary":       {"routing": "bary",       "use_gru": True,  "allow_revise": True,  "hybrid": False},
    "jepa_bary_nogru": {"routing": "bary",       "use_gru": False, "allow_revise": True,  "hybrid": False},
}


def run_experiment(name, cfg, task_id, tr_loader, te_loader, vocab_size,
                   run_idx=0, total_runs=1, global_t0=None):
    hdr = f"  {name.upper()} — Task {task_id}  [{run_idx+1}/{total_runs}]"
    print(f"\n{'=' * 62}\n{hdr}\n{'=' * 62}")

    seed_everything()
    model = RAREJEPA(vocab_size, routing=cfg["routing"],
                     use_gru=cfg["use_gru"],
                     allow_revise=cfg.get("allow_revise", True),
                     hybrid=cfg.get("hybrid", False),
                     ccv_learned=cfg.get("ccv_learned", True)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sclr  = _amp_scaler(USE_AMP)

    hist = defaultdict(list)
    underused = []  # updated each epoch
    epoch_times = []

    for ep in range(TOTAL_EPOCHS):
        phase, eps = get_phase_eps(cfg["routing"], ep)
        t0 = time.time()

        trm = train_epoch(model, tr_loader, opt, sclr, phase, eps, underused)
        tem = evaluate(model, te_loader, phase)
        dt  = time.time() - t0
        epoch_times.append(dt)

        underused = trm.get("underused", [])

        # Record scalar metrics
        for k, v in trm.items():
            if isinstance(v, (int, float)):
                hist[f"tr_{k}"].append(v)
        for k, v in tem.items():
            if isinstance(v, (int, float)):
                hist[f"te_{k}"].append(v)
        hist["te_heatmap"].append(tem.get("heatmap", np.zeros((N_EXPERTS, MAX_STEPS))))
        hist["te_jerr_expert"].append(tem.get("jepa_errs_per_expert", {}))
        hist["te_jerr_per_step"].append(tem.get("jepa_errs_per_step", {}))

        # ETA computation
        avg_ep = np.mean(epoch_times)
        remain_this = (TOTAL_EPOCHS - ep - 1) * avg_ep
        remain_runs = (total_runs - run_idx - 1) * TOTAL_EPOCHS * avg_ep
        eta_s = remain_this + remain_runs
        eta_m, eta_sec = int(eta_s // 60), int(eta_s % 60)
        elapsed = time.time() - global_t0 if global_t0 else 0
        el_m = int(elapsed // 60)

        dist_s = " ".join(f"{d:.2f}" for d in trm["expert_dist"])
        ue_s   = ",".join(str(u) for u in underused) if underused else "-"
        print(f"E{ep:02d} P{phase} ε{eps:.2f} | "
              f"loss {trm['loss']:.4f} cls {trm['L_cls']:.4f} "
              f"jepa {trm.get('L_jepa', 0):.4f} val {trm.get('L_value', 0):.4f} | "
              f"acc {trm['acc']:.1f}/{tem['acc']:.1f}% | "
              f"dist [{dist_s}] ue=[{ue_s}] | "
              f"halt {trm['avg_halt']:.1f} | "
              f"jerr {tem.get('jepa_error_mean', 0):.4f} | "
              f"{dt:.1f}s | ETA {eta_m}m{eta_sec:02d}s (elapsed {el_m}m)")

    return dict(hist)

# ══════════════════════════════════════════════════════════════
# Plotting  (7 figures as specified)
# ══════════════════════════════════════════════════════════════

def plot_all(R):
    tasks = [1, 2, 3]
    exps  = list(EXPERIMENTS.keys())
    C = {"baseline": "#888888", "gumbel": "#e74c3c",
         "jepa": "#2ecc71", "jepa_nogru": "#3498db"}

    # ── (a) Accuracy per task vs epochs ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("(a) Test Accuracy per Task vs Epochs", fontweight="bold")
    for i, t in enumerate(tasks):
        ax = axes[i]
        for e in exps:
            k = f"{e}_t{t}"
            if k in R and "te_acc" in R[k]:
                ax.plot(R[k]["te_acc"], label=e, color=C[e], lw=1.5)
        ax.set_title(f"Task {t}"); ax.set_xlabel("Epoch")
        if i == 0: ax.set_ylabel("Accuracy %")
        ax.axvline(PHASE1_END, color="gray", ls="--", alpha=.4)
        ax.axvline(PHASE2_END, color="gray", ls=":", alpha=.4)
        ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "a_accuracy.png"), dpi=150)

    # ── (b) L_jepa vs epochs ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("(b) JEPA Prediction Loss vs Epochs", fontweight="bold")
    for i, t in enumerate(tasks):
        ax = axes[i]
        for e in ["jepa", "jepa_nogru"]:
            k = f"{e}_t{t}"
            if k in R and "tr_L_jepa" in R[k]:
                ax.plot(R[k]["tr_L_jepa"], label=e, color=C[e], lw=1.5)
        ax.set_title(f"Task {t}"); ax.set_xlabel("Epoch"); ax.set_ylabel("L_jepa")
        ax.axvline(PHASE1_END, color="gray", ls="--", alpha=.4)
        ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "b_jepa_loss.png"), dpi=150)

    # ── (c) Heatmap expert × step (JEPA, final epoch) ────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("(c) Expert Selection Heatmap (JEPA, final epoch)", fontweight="bold")
    for i, t in enumerate(tasks):
        ax = axes[i]
        k = f"jepa_t{t}"
        if k in R and R[k].get("te_heatmap"):
            hm = R[k]["te_heatmap"][-1]
            im = ax.imshow(hm, aspect="auto", cmap="YlOrRd")
            ax.set_xlabel("Step"); ax.set_ylabel("Expert")
            ax.set_title(f"Task {t}")
            ax.set_yticks(range(N_EXPERTS))
            ax.set_xticks(range(MAX_STEPS))
            plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "c_heatmap.png"), dpi=150)

    # ── (d) JEPA prediction error per expert (final epoch) ───
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("(d) JEPA Error per Expert (final epoch)", fontweight="bold")
    for i, t in enumerate(tasks):
        ax = axes[i]
        k = f"jepa_t{t}"
        if k in R and R[k].get("te_jerr_expert"):
            errs = R[k]["te_jerr_expert"][-1]
            vals = [errs.get(e, 0) for e in range(N_EXPERTS)]
            ax.bar(range(N_EXPERTS), vals, color=C["jepa"], alpha=.8)
        ax.set_xlabel("Expert"); ax.set_ylabel("MSE"); ax.set_title(f"Task {t}")
        ax.set_xticks(range(N_EXPERTS))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "d_jepa_error_expert.png"), dpi=150)

    # ── (e) % revise by difficulty (real per-sample data) ──────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("(e) Expert Reuse % by Difficulty (JEPA)", fontweight="bold")
    for i, t in enumerate(tasks):
        ax = axes[i]
        k = f"jepa_t{t}"
        if k in R:
            re = R[k].get("te_revise_easy", [0])[-1] if R[k].get("te_revise_easy") else 0
            rm = R[k].get("te_revise_med",  [0])[-1] if R[k].get("te_revise_med")  else 0
            rh = R[k].get("te_revise_hard", [0])[-1] if R[k].get("te_revise_hard") else 0
            ax.bar(["Easy", "Med", "Hard"], [re, rm, rh],
                   color=["#27ae60", "#f39c12", "#e74c3c"], alpha=.8)
        ax.set_ylabel("% Revise"); ax.set_title(f"Task {t}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "e_revise_difficulty.png"), dpi=150)

    # ── (f) Halt step by difficulty ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("(f) Avg Halt Step by Difficulty (JEPA)", fontweight="bold")
    for i, t in enumerate(tasks):
        ax = axes[i]
        k = f"jepa_t{t}"
        if k in R:
            he = R[k].get("te_halt_easy", [0])[-1] if R[k].get("te_halt_easy") else 0
            hm = R[k].get("te_halt_med",  [0])[-1] if R[k].get("te_halt_med")  else 0
            hh = R[k].get("te_halt_hard", [0])[-1] if R[k].get("te_halt_hard") else 0
            ax.bar(["Easy", "Med", "Hard"], [he, hm, hh],
                   color=["#27ae60", "#f39c12", "#e74c3c"], alpha=.8)
        ax.set_ylabel("Avg Halt Step"); ax.set_title(f"Task {t}")
        ax.set_ylim(0, MAX_STEPS)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "f_halt_difficulty.png"), dpi=150)

    # ── (g) Pareto: accuracy vs compute ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("(g) Pareto: Accuracy vs Compute", fontweight="bold")
    for i, t in enumerate(tasks):
        ax = axes[i]
        for e in exps:
            k = f"{e}_t{t}"
            if k in R:
                acc  = R[k].get("te_acc", [0])[-1]
                halt = R[k].get("te_avg_halt", [N_EXPERTS])[-1] if "te_avg_halt" in R[k] else N_EXPERTS
                ax.scatter(halt, acc, color=C[e], s=100, zorder=5)
                ax.annotate(e, (halt, acc), fontsize=7, ha="center", va="bottom")
        ax.set_xlabel("Avg Expert Forwards"); ax.set_ylabel("Accuracy %")
        ax.set_title(f"Task {t}"); ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "g_pareto.png"), dpi=150)

    if LIVE_PLOT:
        plt.show(block=False)
        plt.pause(0.5)

    print(f"\nPlots saved to {PLOT_DIR}/")

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print("RARE-JEPA: Introspective World Model for Expert Routing")
    print("=" * 62)
    t0_all = time.time()

    # Generate bAbI data
    rng = random.Random(SEED)
    tr_data, te_data = {}, {}
    for tid in [1, 2, 3]:
        samples = TASK_GEN[tid](N_TRAIN + N_TEST, rng)
        tr_data[tid] = samples[:N_TRAIN]
        te_data[tid] = samples[N_TRAIN:]

    vocab = build_vocab(list(tr_data.values()) + list(te_data.values()))
    print(f"Vocab: {len(vocab)} tokens")

    results = {}
    total_runs = 3 * len(EXPERIMENTS)  # 3 tasks × 4 experiments
    run_idx = 0
    for tid in [1, 2, 3]:
        tr_ds = BabiDataset(tr_data[tid], vocab)
        te_ds = BabiDataset(te_data[tid], vocab)
        tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn, num_workers=0, pin_memory=True)
        te_ld = DataLoader(te_ds, BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=0, pin_memory=True)

        for ename, ecfg in EXPERIMENTS.items():
            hist = run_experiment(ename, ecfg, tid, tr_ld, te_ld, len(vocab),
                                 run_idx=run_idx, total_runs=total_runs,
                                 global_t0=t0_all)
            results[f"{ename}_t{tid}"] = hist
            run_idx += 1

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("FINAL TEST ACCURACY %")
    print(f"{'=' * 72}")
    print(f"{'Experiment':<14} {'Task1':>8} {'Task2':>8} {'Task3':>8} {'Avg':>8}")
    print("-" * 46)
    for e in EXPERIMENTS:
        aa = []
        for t in [1, 2, 3]:
            a = results.get(f"{e}_t{t}", {}).get("te_acc", [0])[-1]
            aa.append(a)
        print(f"{e:<14} {aa[0]:>7.1f}% {aa[1]:>7.1f}% {aa[2]:>7.1f}% "
              f"{np.mean(aa):>7.1f}%")

    # ── Kill criteria ─────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("KILL CRITERIA CHECK")
    print(f"{'=' * 62}")
    for t in [1, 2, 3]:
        ja  = results.get(f"jepa_t{t}", {}).get("te_acc", [0])[-1]
        ga  = results.get(f"gumbel_t{t}", {}).get("te_acc", [0])[-1]
        jna = results.get(f"jepa_nogru_t{t}", {}).get("te_acc", [0])[-1]
        jl  = results.get(f"jepa_t{t}", {}).get("tr_L_jepa", [1, 1])
        dist = results.get(f"jepa_t{t}", {}).get("tr_expert_dist", [np.ones(N_EXPERTS)/N_EXPERTS])
        print(f"\nTask {t}:")
        ok1 = "YES" if ja > ga else "NO"
        print(f"  JEPA > Gumbel?     {ok1}  ({ja:.1f}% vs {ga:.1f}%)")
        ok2 = "YES" if len(jl) > 1 and jl[-1] < jl[0] else "NO"
        print(f"  L_jepa descending? {ok2}  ({jl[0]:.4f} -> {jl[-1]:.4f})")
        ok3 = "YES" if ja > jna else "NO"
        print(f"  JEPA > noGRU?      {ok3}  ({ja:.1f}% vs {jna:.1f}%)")

    total_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_min:.1f} min")

    # Save run
    fn = save_run("full_suite", {
        "results": results,
        "total_min": total_min,
    })
    print(f"Run saved: {fn}")

    plot_all(results)


if __name__ == "__main__":
    main()
