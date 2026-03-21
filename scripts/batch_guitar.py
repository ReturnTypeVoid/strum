"""
Batch guitar/bass chart generation for test songs.

Separates stems with Demucs, transcribes guitar and bass,
generates MIDI charts with all 4 difficulty levels.
"""

import logging
import numpy as np
import torch
import librosa
import soundfile as sf
import mido
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import our pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.guitar_bass import (
    transcribe_guitar, transcribe_bass, reduce_to_difficulty, FRET_NOTE_OFFSETS
)


def separate_stems(song_path: Path, output_dir: Path) -> dict:
    """Run Demucs htdemucs stem separation."""
    stems_dir = output_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    # Check if stems already exist
    expected = ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]
    if all((stems_dir / s).exists() for s in expected):
        logger.info(f"  Stems already exist, skipping separation")
        return {s.replace(".wav", ""): stems_dir / s for s in expected}

    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    logger.info(f"  Running Demucs separation...")
    model = get_model("htdemucs")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y, sr_orig = librosa.load(str(song_path), sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])

    target_sr = model.samplerate
    if sr_orig != target_sr:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)

    waveform = torch.from_numpy(y).float()
    ref = waveform.mean(0)
    waveform_norm = (waveform - ref.mean()) / ref.std()
    sources = apply_model(model, waveform_norm[None].to(device), device=device)[0]
    sources = sources * ref.std() + ref.mean()

    stem_paths = {}
    for i, name in enumerate(model.sources):
        stem = sources[i].cpu().numpy()
        path = stems_dir / f"{name}.wav"
        sf.write(str(path), stem.T, target_sr)
        stem_paths[name] = path

    logger.info(f"  Saved {len(stem_paths)} stems")
    return stem_paths


def generate_midi(
    guitar_chart, bass_chart, output_path: Path, tempo_bpm: float
):
    """Generate combined MIDI with guitar and bass tracks."""
    ticks_per_beat = 480
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    # Tempo track
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.name = "Tempo Track"
    tempo_us = int(60_000_000 / tempo_bpm)
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    tempo_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tempo_track.append(mido.MetaMessage("end_of_track", time=0))

    # Events track
    events_track = mido.MidiTrack()
    mid.tracks.append(events_track)
    events_track.name = "EVENTS"
    events_track.append(mido.MetaMessage("end_of_track", time=0))

    for chart, track_name in [(guitar_chart, "PART GUITAR"), (bass_chart, "PART BASS")]:
        if chart is None or not chart.notes:
            continue

        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=track_name, time=0))

        ms_per_tick = (60_000 / tempo_bpm) / ticks_per_beat
        events = []

        for difficulty in ["expert", "hard", "medium", "easy"]:
            diff_chart = reduce_to_difficulty(chart, difficulty)
            note_offset = FRET_NOTE_OFFSETS[difficulty]

            for note in diff_chart.notes:
                start_tick = int(note.time_ms / ms_per_tick)
                duration_ticks = max(int(note.duration_ms / ms_per_tick), ticks_per_beat // 8)
                midi_note = note_offset + note.fret

                events.append(("on", start_tick, midi_note, note.velocity))
                events.append(("off", start_tick + duration_ticks, midi_note, 0))

        events.sort(key=lambda e: (e[1], e[0] == "off"))

        prev_tick = 0
        for etype, tick, note, vel in events:
            delta = max(0, tick - prev_tick)
            track.append(mido.Message(
                'note_on' if etype == 'on' else 'note_off',
                note=note, velocity=vel, time=delta
            ))
            prev_tick = tick

        track.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(track)

    mid.save(str(output_path))


def process_song(song_path: Path, output_dir: Path) -> dict:
    """Process a single song: separate, transcribe, generate MIDI."""
    song_name = song_path.stem
    song_out = output_dir / song_name
    song_out.mkdir(parents=True, exist_ok=True)

    result = {"name": song_name, "success": False}

    try:
        # Step 1: Separate stems
        stems = separate_stems(song_path, song_out)

        # Step 2: Detect tempo from the other stem
        y_other, sr = librosa.load(str(stems["other"]), sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y_other, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        if tempo < 60:
            tempo *= 2
        elif tempo > 200:
            tempo /= 2
        logger.info(f"  Tempo: {tempo:.1f} BPM")

        # Step 3: Transcribe guitar
        guitar = transcribe_guitar(stems["other"], tempo_bpm=tempo)
        result["guitar_notes"] = len(guitar.notes)
        result["guitar_chords"] = len(guitar.chords)
        result["guitar_hopos"] = sum(1 for n in guitar.notes if n.is_hopo)

        # Step 4: Transcribe bass
        bass = transcribe_bass(stems["bass"], tempo_bpm=tempo)
        result["bass_notes"] = len(bass.notes)

        # Step 5: Generate MIDI
        midi_path = song_out / "notes.mid"
        generate_midi(guitar, bass, midi_path, tempo)

        result["success"] = True
        result["tempo"] = tempo
        logger.info(f"  Done: guitar={len(guitar.notes)} bass={len(bass.notes)} -> {midi_path}")

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch guitar/bass chart generation")
    parser.add_argument("--songs-dir", type=Path, default=Path("/mnt/ml-data/sample-songs"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/ml-data/guitar-charts-v1"))
    parser.add_argument("--max-songs", type=int, default=None)
    args = parser.parse_args()

    songs = sorted(args.songs_dir.glob("*.mp3"))
    if args.max_songs:
        songs = songs[:args.max_songs]

    logger.info(f"Processing {len(songs)} songs -> {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, song in enumerate(songs):
        logger.info(f"\n[{i+1}/{len(songs)}] {song.stem}")
        result = process_song(song, args.output_dir)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Song':<45} {'Guitar':>7} {'Bass':>6} {'HOPOs':>6} {'BPM':>5}")
    print("-" * 70)
    for r in results:
        if r["success"]:
            print(f"{r['name'][:44]:<45} {r['guitar_notes']:>7} {r['bass_notes']:>6} "
                  f"{r['guitar_hopos']:>6} {r['tempo']:>5.0f}")
        else:
            print(f"{r['name'][:44]:<45} {'FAILED':>7}")
    print("=" * 70)

    success = sum(1 for r in results if r["success"])
    print(f"\n{success}/{len(results)} songs processed successfully")


if __name__ == "__main__":
    main()
