"""Causal frame indices for compact route training."""
import numpy as np


def build_history(episode, frame, source, length=4):
    """Map each anchor to real earlier observations, without crossing reset boundaries."""
    n = len(episode)
    history = np.repeat(np.arange(n)[:, None], length, axis=1)
    mask = np.zeros((n, length), dtype=bool)
    ages = np.zeros((n, length), dtype=np.float32)
    mask[:, -1] = True
    for ep in np.unique(episode):
        indices = np.flatnonzero((episode == ep) & (source == 0))
        frames = frame[indices]
        if not np.all(np.diff(frames) > 0):
            raise ValueError('Expert frames must be strictly ordered within episode')
        for slot in range(length-1):
            targets = frames - 15 * (length-1-slot)
            previous = np.searchsorted(frames, targets, side='right') - 1
            valid = previous >= 0
            anchors, old = indices[valid], indices[previous[valid]]
            history[anchors, slot] = old
            ages[anchors, slot] = frame[anchors] - frame[old]
            mask[anchors, slot] = True
    assert np.all(episode[history] == episode[:, None])
    assert np.all(frame[history] <= frame[:, None])
    assert np.all(mask[source == 1].sum(1) == 1)
    return history, ages, mask

