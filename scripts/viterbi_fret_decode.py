"""Viterbi decoder for fret-subset sequences.

Takes per-onset fret probabilities P (N, 5) from the MLP mapper and
decodes a sequence of fret subsets that maximizes
    sum_i log p_emit(s_i | P_i)  -  sum_{i>0} trans_cost(s_{i-1}, s_i, dt_i)

State space: all single-fret states (5) + all adjacent 2-fret pairs (4) +
all 3-fret runs (3) + a curated set of common chord shapes. Limiting the
state space prevents the decoder from emitting nonsense like "GRYO".
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _build_state_space() -> list[tuple[int, ...]]:
    """Curated state space — what real charts actually use."""
    states: list[tuple[int, ...]] = []
    # singles
    for i in range(5):
        states.append((i,))
    # adjacent power chords
    for i in range(4):
        states.append((i, i + 1))
    # all 2-subsets (broader; fingers can stretch)
    for a, b in combinations(range(5), 2):
        s = (a, b)
        if s not in states:
            states.append(s)
    # adjacent 3-runs
    for i in range(3):
        states.append((i, i + 1, i + 2))
    # common 3-chord shapes (skipped fret triads)
    for a, b, c in [(0, 2, 4), (0, 1, 3), (1, 2, 4), (0, 2, 3), (1, 3, 4)]:
        if (a, b, c) not in states:
            states.append((a, b, c))
    return states


STATES = _build_state_space()
STATE_VECS = np.zeros((len(STATES), 5), dtype=np.float32)
for i, s in enumerate(STATES):
    for f in s:
        STATE_VECS[i, f] = 1.0


def _emission_logp(P: np.ndarray) -> np.ndarray:
    """log p(state | onset) under independent-Bernoulli model.

    P: (N, 5) probabilities. Returns (N, S).
    """
    P = np.clip(P, 1e-4, 1 - 1e-4)
    logP = np.log(P)
    log1mP = np.log1p(-P)
    # (N, 1, 5) * (1, S, 5) → (N, S)
    return (logP[:, None, :] * STATE_VECS[None, :, :]
            + log1mP[:, None, :] * (1.0 - STATE_VECS[None, :, :])).sum(-1)


def _transition_cost_matrix(
    fret_change_w: float = 0.35,
    center_jump_w: float = 0.05,
    chord_switch_w: float = 0.20,
    same_state_bonus: float = 0.10,
) -> np.ndarray:
    """Cost matrix C[a, b] (subtracted from score)."""
    S = len(STATES)
    centers = np.array([np.mean(s) for s in STATES])
    sizes = np.array([len(s) for s in STATES])
    C = np.zeros((S, S), dtype=np.float32)
    for a in range(S):
        for b in range(S):
            if a == b:
                C[a, b] = -same_state_bonus  # bonus for sticking
                continue
            # number of frets that differ (symmetric difference)
            diff = int(np.abs(STATE_VECS[a] - STATE_VECS[b]).sum())
            C[a, b] += fret_change_w * diff
            C[a, b] += center_jump_w * abs(centers[a] - centers[b])
            # penalty for switching between single and chord (and chord-size changes)
            if sizes[a] != sizes[b]:
                C[a, b] += chord_switch_w * abs(int(sizes[a]) - int(sizes[b]))
    return C


def viterbi_decode(
    P: np.ndarray,
    times: np.ndarray | None = None,
    fret_change_w: float = 0.35,
    center_jump_w: float = 0.05,
    chord_switch_w: float = 0.20,
    same_state_bonus: float = 0.10,
    time_decay_tau: float = 0.5,
) -> list[tuple[int, ...]]:
    """Decode fret-subset sequence from per-onset probabilities.

    Args:
        P: (N, 5) per-onset, per-fret probabilities.
        times: optional (N,) onset times in seconds. If given, transition
            penalties decay with time gap (long pauses = cheap to switch).
        time_decay_tau: seconds; transition cost is multiplied by
            exp(-dt / time_decay_tau).
    """
    N = len(P)
    if N == 0:
        return []
    S = len(STATES)
    E = _emission_logp(P.astype(np.float32))
    C = _transition_cost_matrix(fret_change_w, center_jump_w,
                                chord_switch_w, same_state_bonus)
    # Trellis
    delta = np.full((N, S), -np.inf, dtype=np.float32)
    psi = np.zeros((N, S), dtype=np.int32)
    delta[0] = E[0]
    for i in range(1, N):
        if times is not None:
            dt = float(times[i] - times[i - 1])
            decay = float(np.exp(-max(dt, 0.0) / max(time_decay_tau, 1e-3)))
        else:
            decay = 1.0
        # score[a, b] = delta[i-1, a] - decay * C[a, b]
        scores = delta[i - 1][:, None] - decay * C  # (S, S)
        psi[i] = np.argmax(scores, axis=0)
        delta[i] = E[i] + scores[psi[i], np.arange(S)]
    # Backtrace
    path = np.zeros(N, dtype=np.int32)
    path[-1] = int(np.argmax(delta[-1]))
    for i in range(N - 1, 0, -1):
        path[i - 1] = psi[i, path[i]]
    return [STATES[j] for j in path]


if __name__ == "__main__":
    # Quick smoke
    rng = np.random.default_rng(0)
    P = rng.uniform(size=(10, 5)).astype(np.float32)
    P[3] = [0.9, 0.85, 0.1, 0.1, 0.1]  # power chord
    out = viterbi_decode(P, times=np.arange(10) * 0.2)
    for i, s in enumerate(out):
        print(i, s, P[i].round(2))
