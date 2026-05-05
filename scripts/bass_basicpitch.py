"""bass_basicpitch.py — Basic Pitch + rule-based bass charter.

Bass is monophonic (or near-monophonic), low-register, clean tone — Basic Pitch
handles it well. We map detected pitches to a 5-fret bass chart via per-song
quintile bins on pitch class.

Usage:
  python scripts/bass_basicpitch.py \
      --stems-root /mnt/ml-data/benchmark-pred-v3p1 \
      --out-dir    /mnt/ml-data/benchmark-pred-bass-bp
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import mido

logging.getLogger("root").setLevel(logging.ERROR)
from basic_pitch.inference import predict  # noqa: E402

log = logging.getLogger("bass_basicpitch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FRET_TO_MIDI = {0: 96, 1: 97, 2: 98, 3: 99, 4: 100}  # G,R,Y,B,O
CHORD_GROUP_SEC = 0.030
DEFAULT_NOTE_DUR_SEC = 0.05
DEFAULT_TEMPO_BPM = 120.0


def group_notes_by_onset(notes, window_sec: float = CHORD_GROUP_SEC):
    notes = sorted(notes, key=lambda n: n[0])
    onsets: list[dict] = []
    cur: dict | None = None
    cur_t = -1e9
    for start, end, pitch, amp, _ in notes:
        if start - cur_t > window_sec:
            if cur is not None:
                onsets.append(cur)
            cur = {"time": float(start), "pitches": [], "durations": [], "amps": []}
            cur_t = float(start)
        cur["pitches"].append(int(pitch))
        cur["durations"].append(float(end - start))
        cur["amps"].append(float(amp))
    if cur is not None:
        onsets.append(cur)
    return onsets


def build_pitch_to_fret(all_pitches: list[int]) -> dict[int, int]:
    """Map pitch-class to fret 0–4 via quintile bins on song's PC distribution.

    Bass is monophonic; we use pitch-class so that octave doublings (root +
    octave-up bass slap, harmonic overtones) collapse to the same fret.
    """
    if not all_pitches:
        return {}
    arr = np.asarray(all_pitches)
    pcs = arr % 12
    pc_to_mean = {pc: float(arr[pcs == pc].mean()) for pc in set(pcs.tolist())}
    sorted_pcs = sorted(pc_to_mean.keys(), key=lambda pc: pc_to_mean[pc])
    n = len(sorted_pcs)
    pc_to_fret = {pc: min(4, int(i * 5 / max(1, n))) for i, pc in enumerate(sorted_pcs)}
    return {int(p): pc_to_fret[int(p) % 12] for p in set(arr.tolist())}


def onsets_to_chart(
    onsets: list[dict],
    pitch_to_fret: dict[int, int],
    global_amp_pct: float = 20.0,
    all_amps: list[float] | None = None,
    target_nps: float = 3.5,
    duration_s: float | None = None,
):
    """Pick loudest pitch per onset (bass is monophonic), map to one fret.

    Drop onsets whose loudest pitch falls below the given percentile of all
    amplitudes in the song (suppresses bleed/ghost-detections).

    If `target_nps` and `duration_s` are given AND the resulting onset density
    exceeds 1.5× target, the amplitude percentile is auto-raised until density
    falls into range. Bass charts typically have ≤3-4 notes/sec; sustained
    bleed produces 2-3× that.
    """
    def _filter(amp_pct: float):
        floor = float(np.percentile(np.asarray(all_amps), amp_pct)) if all_amps else 0.0
        out = []
        for ons in onsets:
            if not ons["pitches"]:
                continue
            idx = int(np.argmax(ons["amps"]))
            if ons["amps"][idx] < floor:
                continue
            p = ons["pitches"][idx]
            if p not in pitch_to_fret:
                continue
            out.append((float(ons["time"]), (pitch_to_fret[p],), float(ons["durations"][idx])))
        return out

    events = _filter(global_amp_pct)
    if duration_s and duration_s > 0 and target_nps > 0:
        nps = len(events) / duration_s
        amp_pct = global_amp_pct
        # raise floor in 5pt steps until density is acceptable, capped at p70
        while nps > target_nps * 1.5 and amp_pct < 70.0:
            amp_pct += 5.0
            events = _filter(amp_pct)
            nps = len(events) / duration_s
    return events


def write_chart_midi(
    events: list[tuple[float, tuple[int, ...], float]],
    output_path: Path,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    note_duration_sec: float = DEFAULT_NOTE_DUR_SEC,
    track_name: str = "PART BASS",
) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name))
    tempo_us = int(60_000_000 / tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us))

    tpb = mid.ticks_per_beat
    sec_to_tick = lambda s: int(round(s * tempo_bpm / 60.0 * tpb))

    msgs: list[tuple[int, bool, int]] = []
    for t, frets, _sus in events:
        on_t = sec_to_tick(t)
        off_t = sec_to_tick(t + note_duration_sec)
        for f in frets:
            note = FRET_TO_MIDI[f]
            msgs.append((on_t, True, note))
            msgs.append((off_t, False, note))
    msgs.sort(key=lambda x: (x[0], x[1]))

    last_t = 0
    for tick, is_on, note in msgs:
        delta = max(0, tick - last_t)
        track.append(mido.Message(
            "note_on" if is_on else "note_off",
            note=note, velocity=100 if is_on else 0, time=delta,
        ))
        last_t = tick

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))


def process_song(
    bass_wav: Path,
    out_midi: Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length: int = 11,
    global_amp_pct: float = 20.0,
    track_name: str = "PART BASS",
):
    t0 = time.time()
    _, _, notes = predict(
        str(bass_wav),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
    )
    bp_time = time.time() - t0

    onsets = group_notes_by_onset(notes)
    all_pitches = [p for o in onsets for p in o["pitches"]]
    all_amps = [a for o in onsets for a in o["amps"]]
    p2f = build_pitch_to_fret(all_pitches)
    duration_s = max((o["time"] for o in onsets), default=0.0)
    events = onsets_to_chart(
        onsets, p2f, global_amp_pct=global_amp_pct, all_amps=all_amps,
        duration_s=duration_s,
    )

    write_chart_midi(events, out_midi, track_name=track_name)

    log.info(
        f"  {bass_wav.parent.parent.name}: bp={bp_time:.1f}s  "
        f"raw_notes={len(notes)}  onsets={len(events)}  →  {out_midi}"
    )


def transcribe_bass_basicpitch(
    audio_path: Path,
    tempo_bpm: float = 120.0,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length: int = 11,
    global_amp_pct: float = 20.0,
    note_duration_ms: float = 100.0,
):
    """Transcribe a bass stem to a GuitarChart (instrument='bass').

    Drop-in for `batch_pipeline.transcribe_bass` when STRUM_BASS_BACKEND=basicpitch.
    Returns the same `GuitarChart` dataclass used by every other backend so the
    downstream MIDI-export path is unchanged.
    """
    from src.inference.guitar_bass import GuitarChart, GuitarNote

    _, _, notes = predict(
        str(audio_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
    )
    onsets = group_notes_by_onset(notes)
    all_pitches = [p for o in onsets for p in o["pitches"]]
    all_amps = [a for o in onsets for a in o["amps"]]
    p2f = build_pitch_to_fret(all_pitches)
    duration_s = max((o["time"] for o in onsets), default=0.0)
    events = onsets_to_chart(
        onsets, p2f, global_amp_pct=global_amp_pct, all_amps=all_amps,
        duration_s=duration_s,
    )

    chart = GuitarChart(tempo_bpm=float(tempo_bpm), instrument="bass")
    for t, frets, _sus in events:
        # bass spike emits exactly one fret per onset
        chart.notes.append(GuitarNote(
            time_ms=t * 1000.0,
            fret=int(frets[0]),
            duration_ms=note_duration_ms,
        ))
    return chart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--track", default="PART BASS")
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--frame-threshold", type=float, default=0.3)
    ap.add_argument("--min-note-length", type=int, default=11)
    ap.add_argument("--global-amp-pct", type=float, default=20.0)
    ap.add_argument("--stem-name", default="bass.wav")
    args = ap.parse_args()

    songs = sorted([d for d in args.stems_root.iterdir() if d.is_dir()])
    log.info(f"Processing {len(songs)} songs from {args.stems_root}")

    for song_dir in songs:
        bass_wav = song_dir / "stems" / args.stem_name
        if not bass_wav.exists():
            log.warning(f"  SKIP {song_dir.name}: missing {bass_wav}")
            continue
        out_midi = args.out_dir / song_dir.name / "notes.mid"
        try:
            process_song(
                bass_wav, out_midi,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                min_note_length=args.min_note_length,
                global_amp_pct=args.global_amp_pct,
                track_name=args.track,
            )
        except Exception as e:
            log.error(f"  FAIL {song_dir.name}: {e}")


if __name__ == "__main__":
    main()
