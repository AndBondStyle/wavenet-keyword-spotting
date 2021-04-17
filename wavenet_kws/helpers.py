from pathlib import Path
import numpy as np

__all__ = [
    "get_random_file",
    "get_random_chunk",
    "get_weighted_item",
    "merge_signals",
]


def get_random_file(path: Path, glob: str = "*.wav"):
    if not hasattr(get_random_file, "__cache__"):
        setattr(get_random_file, "__cache__", {})
    cache = getattr(get_random_file, "__cache__", {})
    cache_key = (str(path), glob)
    if cache_key not in cache:
        files = list(path.glob(glob))
        cache[cache_key] = files
    files = cache[cache_key]
    ind = np.random.choice(len(files))
    return files[ind]


def get_weighted_item(choices: dict):
    choices = list(choices.items())
    weights = [x[1] for x in choices]
    items = list(range(len(choices)))
    ind = np.random.choice(items, p=weights)
    return choices[ind][0]


def get_random_chunk(signal, length: float, sample_rate: int):
    length = int(length * sample_rate)
    assert len(signal) >= length, f"len(signal) = {len(signal)}, length = {length}"
    start = np.random.randint(0, len(signal) - length)
    return signal[start : start + length]


def merge_signals(*items, sample_rate: int, length: float):
    length = int(length * sample_rate)
    signal = np.zeros(length, dtype="float32")
    for chunk, start, end in items:
        if start is None:
            assert end is not None
            end = int(end * sample_rate)
            start = end - len(chunk)
        else:
            start = int(start * sample_rate)
            end = start + len(chunk)
        if start >= length or end < 0:
            continue
        if start >= 0:
            end = min(end, length)
            chunk = chunk[: end - start]
            signal[start:end] += chunk
        elif end <= length:
            start = max(start, 0)
            chunk = chunk[-(end - start) :]
            signal[start:end] += chunk
        else:
            assert len(chunk) > length
            offset = 0 - start
            chunk = chunk[offset : offset + length]
            signal += chunk
    return signal
