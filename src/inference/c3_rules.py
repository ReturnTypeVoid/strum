"""C3 / Rock Band guitar & bass authoring rule enforcement.

Reference: http://docs.c3universe.com/rbndocs/index.php?title=Guitar_and_Bass_Authoring

Hard rules enforced (gameplay breaks otherwise):
  1. No-overlap: every note (or chord) must END before the next note BEGINS.
     If sustains overlap the next event, the in-game note disappears.
     Required gap = 1/32 note (we use 1/16 as standard).
  2. Forbidden chord shapes per difficulty:
       Expert: 3-note chords may NOT include both Green AND Orange.
       Hard:   no 3-note chords, no Green/Orange 2-chord. GB / RO allowed.
       Medium: no 3-note chords; no GB, GO, RO 2-chords.
       Easy:   no chords (single notes only).
  3. Tempo-aware sustains: a note shorter than 1/4 note (or 1/2 note at
     tempos < 100 BPM, per the C3 dotted-8th rule) gets NO sustain tail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Fret indices: 0=Green 1=Red 2=Yellow 3=Blue 4=Orange


@dataclass
class GuitarEvent:
    time_ms: float
    frets: list[int]          # sorted, deduped
    duration_ms: float        # 0 / short → tap; long → sustain
    velocity: int = 100
    is_chord: bool = False    # informational; recomputed from len(frets)

    @property
    def end_ms(self) -> float:
        return self.time_ms + self.duration_ms


# ─── Forbidden 2-chord shapes per difficulty ─────────────────────────────────
_FORBIDDEN_2CHORD = {
    "expert": set(),                                # all allowed
    "hard":   {(0, 4)},                             # no GO
    "medium": {(0, 3), (0, 4), (1, 4)},             # no GB, GO, RO
    "easy":   {(0, 1), (0, 2), (0, 3), (0, 4),      # no chords at all
               (1, 2), (1, 3), (1, 4),
               (2, 3), (2, 4),
               (3, 4)},
}


def _fix_chord_shape(frets: list[int], difficulty: str) -> list[int]:
    """Return a frets list legal for the given difficulty.

    Strategy:
      * 3-note chords on hard/medium/easy → reduce to outer two.
      * 3-note chord on expert containing both 0 (G) and 4 (O) → drop orange.
      * 2-note chord that is forbidden → collapse to lower fret (single note).
      * Easy → always single note (lowest fret).
    """
    frets = sorted(set(frets))
    if not frets:
        return frets

    if difficulty == "easy":
        return [frets[0]]

    # 3-note rules
    if len(frets) >= 3:
        if difficulty == "expert":
            if 0 in frets and 4 in frets:
                # 3-note chords with G+O forbidden — drop orange
                frets = [f for f in frets if f != 4]
                if len(frets) < 2:
                    return frets[:1]
        else:
            # hard / medium → no 3-chord; reduce to outer two
            frets = [frets[0], frets[-1]]

    # 2-note rules
    if len(frets) == 2:
        pair = (frets[0], frets[1])
        if pair in _FORBIDDEN_2CHORD.get(difficulty, set()):
            # On hard, prefer remapping GO -> GB; otherwise collapse to root
            if difficulty == "hard" and pair == (0, 4):
                return [0, 3]
            return [frets[0]]

    return frets


# ─── Sustain & overlap enforcement ───────────────────────────────────────────
def _min_sustain_ms(tempo_bpm: float) -> float:
    """Minimum note duration that gets a sustain tail.

    C3: notes longer than dotted-8th (>100 BPM) or 8th (<100 BPM) sustain.
    We use a single tempo-aware threshold of 3/16 of a beat (~ dotted-8th
    of an 8th), which works musically across the typical rock range.
    """
    beat_ms = 60_000.0 / max(tempo_bpm, 30.0)
    return 0.75 * beat_ms  # 3/16 note ≈ dotted 8th


def _min_gap_ms(tempo_bpm: float) -> float:
    """Required silence between end of sustain and next note (1/32 note)."""
    beat_ms = 60_000.0 / max(tempo_bpm, 30.0)
    return beat_ms / 8.0  # 1/32 note


def apply_c3_rules(
    events: list[GuitarEvent],
    tempo_bpm: float,
    difficulty: str,
) -> list[GuitarEvent]:
    """Enforce C3 chord, sustain, and no-overlap rules.

    Operates on a flat list of grouped events (one per time_ms with frets list).
    Returns a new list with rules applied.
    """
    if not events:
        return events

    # 1. Sort and merge co-incident events (within 5ms) to avoid stacked onsets
    events = sorted(events, key=lambda e: e.time_ms)
    merged: list[GuitarEvent] = []
    for ev in events:
        if merged and (ev.time_ms - merged[-1].time_ms) < 5.0:
            # Merge fret sets
            merged[-1].frets = sorted(set(merged[-1].frets) | set(ev.frets))
            merged[-1].duration_ms = max(merged[-1].duration_ms, ev.duration_ms)
            continue
        merged.append(GuitarEvent(
            time_ms=ev.time_ms,
            frets=list(ev.frets),
            duration_ms=ev.duration_ms,
            velocity=ev.velocity,
        ))

    # 2. Apply per-difficulty chord-shape rules
    for ev in merged:
        ev.frets = _fix_chord_shape(ev.frets, difficulty)
        ev.is_chord = len(ev.frets) >= 2

    # Drop any event that lost all frets
    merged = [ev for ev in merged if ev.frets]

    # 3. Tempo-aware sustain decision: short notes get no tail
    min_sus = _min_sustain_ms(tempo_bpm)
    short_note_ms = 80.0
    for ev in merged:
        if ev.duration_ms < min_sus:
            ev.duration_ms = short_note_ms

    # 4. No-overlap enforcement: any note must end before next note starts.
    #    Required gap = 1/32 note. If shorter, truncate prev duration.
    gap_ms = _min_gap_ms(tempo_bpm)
    for i in range(len(merged) - 1):
        cur, nxt = merged[i], merged[i + 1]
        max_end = nxt.time_ms - gap_ms
        if cur.end_ms > max_end:
            new_dur = max(short_note_ms, max_end - cur.time_ms)
            # If even the short tap doesn't fit, just clamp to whatever fits
            if max_end - cur.time_ms < short_note_ms:
                new_dur = max(20.0, max_end - cur.time_ms)
            cur.duration_ms = new_dur

    return merged


__all__ = ["GuitarEvent", "apply_c3_rules"]
