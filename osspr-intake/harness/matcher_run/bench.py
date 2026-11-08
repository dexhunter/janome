"""Steady-state benchmark for janome.fst.Matcher._run.

Inputs are fixed. PYTHONHASHSEED is pinned so the prefix cache and output
sets iterate in the same order on every run.
"""

import ctypes
import gc
import os
import statistics
import sys
import time

# Pin hash order and virtual addresses so repeated runs see the same layout.
_ADDR_NO_RANDOMIZE = 0x0040000
_persona = ctypes.CDLL(None).personality(0xFFFFFFFF)
_need_persona = (_persona & _ADDR_NO_RANDOMIZE) == 0
_need_seed = os.environ.get("PYTHONHASHSEED") != "0"
if _need_persona or _need_seed:
    if _need_seed:
        os.environ["PYTHONHASHSEED"] = "0"
    if _need_persona:
        ctypes.CDLL(None).personality(_persona | _ADDR_NO_RANDOMIZE)
    os.execv(sys.executable, [sys.executable, *sys.argv])
try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

from janome.tokenizer import Tokenizer

# One repeat walks this many windows. Large enough that the median is stable.
INNER_LOOPS = 56
WARMUP = 3
REPEATS = 11


def build_windows(tokenizer):
    """50-character windows, the same slice the tokenizer encodes at each position."""
    dictionary = tokenizer.sys_dic
    last = max(end for _, end in dictionary.bucket_ranges)
    particles = ("の", "は", "を", "に", "が", "と", "で", "も")
    pieces = []
    for n, idx in enumerate(range(0, last, 197)):
        surface = dictionary._find_entry(idx)[0]
        if not surface:
            continue
        pieces.append(surface)
        pieces.append(particles[n % len(particles)])
        if n >= 699:
            break
    text = "".join(pieces)
    windows = []
    limit = max(1, len(text) - 1)
    for start in range(0, limit, 5):
        windows.append(text[start : start + 50].encode("utf-8"))
    if not windows:
        raise RuntimeError("no benchmark windows")
    return windows


def main():
    tokenizer = Tokenizer(mmap=True)
    matcher = tokenizer.matcher
    windows = build_windows(tokenizer)
    parts = range(matcher.dict_len)
    run = matcher._run
    # The common-prefix search used by dictionary lookup.
    probe = run("すもも".encode("utf-8"), 0, True)
    if not isinstance(probe, set):
        raise RuntimeError("Matcher._run did not return a set")

    def once():
        for _ in range(INNER_LOOPS):
            for word in windows:
                for part in parts:
                    run(word, part, True)

    gc.collect()
    gc.disable()
    try:
        for _ in range(WARMUP):
            once()
        samples = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            once()
            samples.append((time.perf_counter() - started) * 1000.0)
    finally:
        gc.enable()
    print(f"median_ms={statistics.median(samples):.4f}")


if __name__ == "__main__":
    main()
