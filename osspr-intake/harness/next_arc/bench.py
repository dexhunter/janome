"""Steady-state benchmark for janome.fst.Matcher.next_arc.

Addresses come from a fixed dictionary-lookup walk and then repeat in that
order. The working set is larger than next_arc's 4096-entry cache, which is
the miss-heavy regime seen while tokenizing varied text.
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

from janome.fst import Matcher
from janome.tokenizer import Tokenizer

WARMUP = 2
REPEATS = 11
INNER_LOOPS = 64
TARGET_ARCS = 6000


def main():
    tokenizer = Tokenizer(mmap=True)
    matcher = tokenizer.matcher
    raw = Matcher.next_arc.__wrapped__
    recorded = []
    seen = set()
    id_to_part = {id(blob): part for part, blob in enumerate(matcher.dict_data)}
    original = Matcher.next_arc

    def tap(self, data, addr):
        key = (id_to_part[id(data)], addr)
        if key not in seen:
            seen.add(key)
            recorded.append(key)
        return raw(self, data, addr)

    Matcher.next_arc = tap
    try:
        dictionary = tokenizer.sys_dic
        last = max(end for _, end in dictionary.bucket_ranges)
        for idx in range(0, last, 90):
            surface = dictionary._find_entry(idx)[0]
            if not surface:
                continue
            word = surface.encode("utf-8")
            for part in range(matcher.dict_len):
                matcher._run(word, part, True)
            if len(recorded) >= TARGET_ARCS:
                break
    finally:
        Matcher.next_arc = original

    if len(recorded) <= 4096:
        raise RuntimeError(f"need more than 4096 distinct arcs, found {len(recorded)}")

    original.cache_clear()
    blobs = matcher.dict_data
    decode = matcher.next_arc
    sample = decode(blobs[recorded[0][0]], recorded[0][1])
    if len(sample) != 6:
        raise RuntimeError("next_arc did not return a 6-tuple")
    original.cache_clear()

    def once():
        for _ in range(INNER_LOOPS):
            for part, addr in recorded:
                decode(blobs[part], addr)

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
