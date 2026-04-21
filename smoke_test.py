#!/usr/bin/env python3
"""Quick smoke test for RARE-JEPA."""
import sys
sys.path.insert(0, ".")
import rare_jepa as rj
import torch

rj.seed_everything()
V = 50
D = rj.DEVICE
tokens = torch.randint(1, V, (4, 20)).to(D)
mask = torch.ones(4, 20, dtype=torch.bool).to(D)
targets = torch.randint(0, V, (4,)).to(D)

# Test all routing modes
for routing in ["sequential", "gumbel", "jepa"]:
    print(f"\n--- {routing} ---")
    model = rj.RAREJEPA(V, routing=routing, use_gru=True).to(D)
    model.train()

    # Phase 1
    logits, info = model(tokens, mask, phase=1, targets=targets)
    n_steps = len(info["expert_choices"])
    print(f"  Phase1: logits={logits.shape}, steps={n_steps}")

    loss, parts = rj.compute_loss(logits, targets, info, D)
    loss.backward()
    print(f"  Loss={loss.item():.4f} | {parts}")

    # Phase 2 (skip for sequential)
    if routing != "sequential":
        model.zero_grad()
        logits2, info2 = model(tokens, mask, phase=2, epsilon=0.3, targets=targets)
        n2 = len(info2["expert_choices"])
        halts = info2["halt_steps"].tolist()
        print(f"  Phase2: steps={n2}, halts={halts}")
        loss2, parts2 = rj.compute_loss(logits2, targets, info2, D)
        loss2.backward()
        print(f"  Loss={loss2.item():.4f} backward OK")

# Test JEPA without GRU
print("\n--- jepa_nogru ---")
m_ng = rj.RAREJEPA(V, routing="jepa", use_gru=False).to(D)
m_ng.train()
logits_ng, info_ng = m_ng(tokens, mask, phase=2, epsilon=0.1, targets=targets)
loss_ng, _ = rj.compute_loss(logits_ng, targets, info_ng, D)
loss_ng.backward()
print(f"  OK, loss={loss_ng.item():.4f}")

# Test data generation
rng = rj.random.Random(42)
for tid in [1, 2, 3]:
    samples = rj.TASK_GEN[tid](10, rng)
    print(f"\nTask {tid} sample: {samples[0]}")

print("\n=== ALL SMOKE TESTS PASSED ===")
