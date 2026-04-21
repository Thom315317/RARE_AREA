"""
bAbI tasks loader — en-10k from local files.
Downloaded via: curl https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz
"""
import os, re, glob
from typing import List, Tuple
from functools import lru_cache

BABI_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "babi", "tasks_1-20_v1-2", "en-10k")


def _tokenize(text: str) -> str:
    """Split punctuation + lowercase + collapse whitespace."""
    text = text.lower()
    text = re.sub(r"([.,?!])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_babi_file(path: str) -> List[Tuple[str, str, str]]:
    """Parse a bAbI text file into (story, question, answer) triples.

    bAbI format lines:
      "<id> <text>"          → fact sentence (add to story)
      "<id> <q>\t<ans>\t<s>" → question (tab-separated answer + support ids)
    Story resets when <id> == 1.
    """
    samples = []
    story: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sp = line.split(" ", 1)
            if len(sp) < 2:
                continue
            try:
                lid = int(sp[0])
            except ValueError:
                continue
            content = sp[1]
            if lid == 1:
                story = []                  # new story

            if "\t" in content:
                # Question line
                q_text, ans, *_ = content.split("\t")
                story_text = " ".join(story)
                samples.append((
                    _tokenize(story_text),
                    _tokenize(q_text.strip()),
                    ans.strip().lower(),
                ))
            else:
                story.append(content.strip())
    return samples


@lru_cache(maxsize=None)
def load_task(task_id: int) -> Tuple[list, list]:
    """Load train + test for task id (1..20) from en-10k."""
    if not os.path.isdir(BABI_ROOT):
        raise FileNotFoundError(
            f"bAbI en-10k not found at {BABI_ROOT}.\n"
            f"Download: curl https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz "
            f"| tar xz -C data/babi/")
    prefix = f"qa{task_id}_"
    train_file = test_file = None
    for fn in os.listdir(BABI_ROOT):
        if fn.startswith(prefix):
            if fn.endswith("_train.txt"):
                train_file = os.path.join(BABI_ROOT, fn)
            elif fn.endswith("_test.txt"):
                test_file = os.path.join(BABI_ROOT, fn)
    if not train_file or not test_file:
        raise FileNotFoundError(f"Task {task_id} not found in {BABI_ROOT}")
    return parse_babi_file(train_file), parse_babi_file(test_file)


def task_name(task_id: int) -> str:
    names = {
        1: "single-supporting-fact",       2: "two-supporting-facts",
        3: "three-supporting-facts",       4: "two-arg-relations",
        5: "three-arg-relations",          6: "yes-no-questions",
        7: "counting",                     8: "lists-sets",
        9: "simple-negation",             10: "indefinite-knowledge",
        11: "basic-coreference",          12: "conjunction",
        13: "compound-coreference",       14: "time-reasoning",
        15: "basic-deduction",            16: "basic-induction",
        17: "positional-reasoning",       18: "size-reasoning",
        19: "path-finding",               20: "agents-motivations",
    }
    return names.get(task_id, f"task-{task_id}")


if __name__ == "__main__":
    for tid in [1, 6, 19]:
        tr, te = load_task(tid)
        print(f"Task {tid} ({task_name(tid)}): {len(tr)} train, {len(te)} test")
        print(f"  Story: {tr[0][0][:80]}...")
        print(f"  Q: {tr[0][1]} → A: {tr[0][2]}")
