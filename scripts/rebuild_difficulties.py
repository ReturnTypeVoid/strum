#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import mido

from chart_enhancer import ChartEnhancer


logging.basicConfig(level=logging.INFO)


def rebuild_difficulties(midi_path: str, output_path: str):
    midi_path = Path(midi_path)
    output_path = Path(output_path)

    enhancer = ChartEnhancer()

    midi = mido.MidiFile(midi_path)
    enhancer.ticks_per_beat = midi.ticks_per_beat

    # Get tempo
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                enhancer.tempo_bpm = round(mido.tempo2bpm(msg.tempo))
                break

    enhancer.ticks_per_sec = (
        enhancer.ticks_per_beat * enhancer.tempo_bpm / 60
    )

    logging.info(
        f"Tempo: {enhancer.tempo_bpm} BPM, "
        f"Resolution: {enhancer.ticks_per_beat}"
    )

    new_tracks = []

    for track in midi.tracks:
        track_name = None

        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.name
                break

        if track_name in ['PART GUITAR', 'PART BASS']:
            logging.info(f"Rebuilding difficulties: {track_name}")
            new_tracks.append(
                enhancer.apply_difficulty_reduction(track)
            )

        elif track_name == 'PART DRUMS':
            logging.info("Rebuilding difficulties: PART DRUMS")
            new_tracks.append(
                enhancer.apply_drums_difficulty_reduction(track)
            )

        elif track_name == 'PART KEYS':
            logging.info("Rebuilding difficulties: PART KEYS")
            new_tracks.append(
                enhancer.apply_keys_difficulty_reduction(track)
            )

        else:
            new_tracks.append(track)

    midi.tracks = new_tracks
    midi.save(output_path)

    logging.info(f"Saved: {output_path}")

    print_difficulty_stats(output_path)


def print_difficulty_stats(midi_path):
    import collections

    midi = mido.MidiFile(midi_path)

    print("\n[OCTAVE] DIFFICULTY RESULT")

    for track in midi.tracks:
        track_name = None

        for msg in track:
            if msg.type == "track_name":
                track_name = msg.name
                break

        if track_name not in [
            "PART GUITAR",
            "PART BASS",
            "PART KEYS",
            "PART DRUMS",
        ]:
            continue

        counts = collections.Counter()

        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                if 96 <= msg.note <= 100:
                    counts["Expert"] += 1
                elif 84 <= msg.note <= 88:
                    counts["Hard"] += 1
                elif 72 <= msg.note <= 76:
                    counts["Medium"] += 1
                elif 60 <= msg.note <= 64:
                    counts["Easy"] += 1

        expert = max(counts["Expert"], 1)

        print(track_name)
        print(
            f"  Expert: {counts['Expert']}"
        )
        print(
            f"  Hard:   {counts['Hard']} "
            f"({counts['Hard']/expert:.0%})"
        )
        print(
            f"  Medium: {counts['Medium']} "
            f"({counts['Medium']/expert:.0%})"
        )
        print(
            f"  Easy:   {counts['Easy']} "
            f"({counts['Easy']/expert:.0%})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild Clone Hero difficulties without running Octave"
    )

    parser.add_argument(
        "midi",
        help="Input notes.mid"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output MIDI (default: notes-difficulties.mid)"
    )

    args = parser.parse_args()

    output = args.output or (
        str(Path(args.midi).with_name("notes-difficulties.mid"))
    )

    rebuild_difficulties(args.midi, output)


if __name__ == "__main__":
    main()
