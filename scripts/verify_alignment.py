#!/usr/bin/env python3
"""Verify alignment: after applying audio_offset_ms, residual offsets should be ~0.

Re-runs the same offset sweep used during alignment, but treats the corrected
GT (midi_time - audio_offset_ms) as ground truth and measures how much
additional shift would still help. A correctly aligned dataset should have
median residual offset ~0 ms.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, yaml
from scipy.signal import find_peaks
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.models.guitar_v1 import GuitarOnsetCRNN, OnsetCRNNConfig  # noqa
import preprocess_guitar_windows as pgw  # noqa
import eval_guitar_onset as ego  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/guitar_v1_manifest_aligned.json")
    ap.add_argument("--checkpoint", default="checkpoints/guitar_v1/guitar_v1_onset/best.pt")
    ap.add_argument("--config", default="configs/guitar_v1.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--sweep-ms", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sr = cfg["audio"]["sample_rate"]; hop = cfg["audio"]["hop_length"]
    frame_s = hop / sr

    model = GuitarOnsetCRNN(OnsetCRNNConfig(**cfg["onset"]["model"])).to(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    mel_extractor = pgw.get_mel()

    manifest = json.load(open(args.manifest))
    songs = [s for s in manifest["songs"] if s["split"] == args.split]
    if args.limit > 0: songs = songs[:args.limit]
    print(f"verifying {len(songs)} {args.split} songs from {args.manifest}")

    rows = []
    with torch.no_grad():
        for s in tqdm(songs, desc="verify"):
            try:
                y = ego.load_audio(Path(s["audio_path"]), sr=sr)
                mel = ego.compute_full_logmel(y, mel_extractor).to(args.device).unsqueeze(0)
                probs = torch.sigmoid(model(mel)).squeeze(0).cpu().numpy()
                events = pgw.parse_onsets_from_manifest(Path(s["midi_path"]))
                if not events: continue
                # Apply correction (same logic as preprocess)
                off = float(s.get("audio_offset_ms", 0) or 0)
                gt = np.array([(t - off) / 1000.0 for (t, _) in events])
                gt = gt[gt >= 0]
                if not len(gt): continue
                peaks, _ = find_peaks(probs, height=0.10, distance=1)
                pred = peaks * frame_s
                if not len(pred): continue
                # Sweep residual
                best_r, best_off = -1, 0
                for off_ms in range(-args.sweep_ms, args.sweep_ms + 1, 5):
                    tp,_,fn = ego.onset_f1(pred + off_ms/1000.0, gt, 0.05)
                    r = tp/max(tp+fn,1)
                    if r > best_r: best_r, best_off = r, off_ms
                tp,_,fn = ego.onset_f1(pred, gt, 0.05); r0 = tp/max(tp+fn,1)
                rows.append((s["id"], s.get("align_status"), off, r0, best_r, best_off))
            except Exception:
                pass

    if not rows:
        print("no rows"); return
    residual = np.array([r[5] for r in rows])
    r0s = np.array([r[3] for r in rows])
    rbest = np.array([r[4] for r in rows])
    print(f"\n=== verification on {len(rows)} songs ===")
    print(f"residual |offset|:  median={np.median(np.abs(residual)):.0f}ms  "
          f"p75={np.percentile(np.abs(residual),75):.0f}ms  "
          f"p90={np.percentile(np.abs(residual),90):.0f}ms  "
          f"max={np.abs(residual).max()}ms")
    print(f"|residual| > 50 ms: {(np.abs(residual)>50).sum()}/{len(residual)}  "
          f"(was 54/100 before alignment)")
    print(f"recall @ off=0:  median={np.median(r0s):.3f}  mean={r0s.mean():.3f}")
    print(f"recall @ best:   median={np.median(rbest):.3f}  mean={rbest.mean():.3f}")
    print(f"residual gain:   mean={(rbest-r0s).mean():+.3f}  (smaller is better)")

    # By status
    import collections
    by_st = collections.defaultdict(list)
    for sid, st, off, r0, rb, res in rows:
        by_st[st].append((r0, rb, res))
    for st, vals in by_st.items():
        r0s2 = np.array([v[0] for v in vals])
        ress2 = np.array([abs(v[2]) for v in vals])
        print(f"  {st:<18} n={len(vals):>3}  median residual={np.median(ress2):.0f}ms  mean R@0={r0s2.mean():.3f}")


if __name__ == "__main__":
    main()
