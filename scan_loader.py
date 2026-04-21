"""
SCAN dataset loader — compositional generalization benchmark.
Source: https://github.com/brendenlake/SCAN

Downloads raw .txt files from GitHub, parses "IN: ... OUT: ..." lines
into (command, actions) pairs.

Splits supported:
  simple        : random 80/20 (control)
  length        : train short, test long
  addprim_jump  : train with 'jump' in simple contexts, test in complex
  addprim_turn_left : same with 'turn left'
  template*     : template-based splits
"""
import os, re, urllib.request
from functools import lru_cache
from typing import List, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "scan")

# Mapping split name → (train_url, test_url)
BASE = "https://raw.githubusercontent.com/brendenlake/SCAN/master"
SCAN_URLS = {
    "simple": (
        f"{BASE}/simple_split/tasks_train_simple.txt",
        f"{BASE}/simple_split/tasks_test_simple.txt",
    ),
    "length": (
        f"{BASE}/length_split/tasks_train_length.txt",
        f"{BASE}/length_split/tasks_test_length.txt",
    ),
    "addprim_jump": (
        f"{BASE}/add_prim_split/tasks_train_addprim_jump.txt",
        f"{BASE}/add_prim_split/tasks_test_addprim_jump.txt",
    ),
    "addprim_turn_left": (
        f"{BASE}/add_prim_split/tasks_train_addprim_turn_left.txt",
        f"{BASE}/add_prim_split/tasks_test_addprim_turn_left.txt",
    ),
}


def _download(url: str, dest: str):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)


def parse_scan_file(path: str) -> List[Tuple[str, str]]:
    """Parse SCAN file into list of (command, actions) tuples."""
    pairs = []
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"IN:\s*(.*?)\s*OUT:\s*(.*)$", line.strip())
            if m:
                cmd = m.group(1).strip()
                act = m.group(2).strip()
                pairs.append((cmd, act))
    return pairs


@lru_cache(maxsize=None)
def load_split(split_name: str) -> Tuple[list, list]:
    """Load (train, test) for a SCAN split."""
    if split_name not in SCAN_URLS:
        raise ValueError(f"Unknown split: {split_name}. "
                         f"Available: {list(SCAN_URLS)}")
    train_url, test_url = SCAN_URLS[split_name]
    train_path = os.path.join(DATA_DIR, split_name, os.path.basename(train_url))
    test_path  = os.path.join(DATA_DIR, split_name, os.path.basename(test_url))
    _download(train_url, train_path)
    _download(test_url, test_path)
    return parse_scan_file(train_path), parse_scan_file(test_path)


if __name__ == "__main__":
    for sp in ["simple", "length", "addprim_jump"]:
        tr, te = load_split(sp)
        print(f"Split {sp}: {len(tr)} train / {len(te)} test")
        print(f"  Example: IN='{tr[0][0]}'  OUT='{tr[0][1]}'")
        # Get vocabs
        in_vocab = set()
        out_vocab = set()
        for c, a in tr + te:
            in_vocab.update(c.split())
            out_vocab.update(a.split())
        print(f"  Input vocab: {len(in_vocab)}  Output vocab: {len(out_vocab)}")
        max_in = max(len(c.split()) for c, _ in tr + te)
        max_out = max(len(a.split()) for _, a in tr + te)
        print(f"  Max lengths: in={max_in}  out={max_out}")
