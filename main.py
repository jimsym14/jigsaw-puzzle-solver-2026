"""
main.py — CLI runner για το Jigsaw Puzzle Solver.

Χρήση:
  python main.py --image I2.jpg --seed 42
  python main.py --all --seed 42
  python main.py --all --seed 42 --clean-results
"""

import os
import sys
import shutil
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from puzzle_utils import INPUT_DIR, RESULTS_DIR, get_stage_paths, run_stage1
from features import run_stage2
from solver import run_stage3


def run_pipeline(image_path, seed=42, run_abl=False):
    """Τρέχει Stage 1→2→3 για μία εικόνα. Επιστρέφει dict με metrics."""
    # Stage 1 — shuffle
    stage1 = run_stage1(image_path, seed=seed, clear_prev=True)
    puzzle  = stage1['puzzle']
    tiles   = stage1['scrambled_tiles']
    paths   = stage1['paths']

    os.makedirs(paths['stage2'], exist_ok=True)
    os.makedirs(paths['stage3'], exist_ok=True)

    # Stage 2 — features
    all_features = run_stage2(tiles, paths['stage2'])

    # Stage 3 — solve + evaluate
    result = run_stage3(tiles, puzzle, all_features, paths['stage3'])

    # Stage 4 — ablation (αν ζητήθηκε)
    if run_abl:
        from ablation import run_ablation
        print(f"\n  ── Ablation Study ──")
        run_ablation(image_path, seed=seed)

    return {
        'image':    stage1['image_name'],
        'pos_acc':  result['pos_acc'],
        'rot_acc':  result['rot_acc'],
        'neigh_acc':result['neigh_acc'],
        'time':     result['time'],
    }


def run_single(image_path, seed, run_abl=False):
    run_pipeline(image_path, seed=seed, run_abl=run_abl)


def run_all(seed, run_abl=False):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts))
    if not images:
        raise FileNotFoundError(f"Δεν βρέθηκαν εικόνες στο {INPUT_DIR}")

    print(f"\n{'═'*55}")
    print(f"BATCH RUN — {len(images)} images | seed={seed}")
    print(f"{'═'*55}")

    results = []
    for i, fname in enumerate(images, 1):
        print(f"\n── {i}/{len(images)}: {fname}")
        r = run_pipeline(os.path.join(INPUT_DIR, fname), seed=seed, run_abl=run_abl)
        results.append(r)

    # Summary
    avg_pos   = np.mean([r['pos_acc']   for r in results])
    avg_rot   = np.mean([r['rot_acc']   for r in results])
    avg_neigh = np.mean([r['neigh_acc'] for r in results])
    total_t   = sum(r['time'] for r in results)

    print(f"\n{'═'*55}")
    print(f"{'Image':<15} {'Pos%':>7} {'Rot%':>7} {'Neigh%':>8} {'Time':>7}")
    print(f"{'-'*46}")
    for r in results:
        print(f"{r['image']:<15} {r['pos_acc']:>6.1f}% {r['rot_acc']:>6.1f}% "
              f"{r['neigh_acc']:>7.1f}% {r['time']:>6.1f}s")
    print(f"{'-'*46}")
    print(f"{'AVERAGE':<15} {avg_pos:>6.1f}% {avg_rot:>6.1f}% "
          f"{avg_neigh:>7.1f}% {total_t:>6.1f}s")

    # Save batch summary txt
    from datetime import datetime
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "batch_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Batch Run — {datetime.now():%Y-%m-%d %H:%M} | seed={seed}\n{'='*55}\n")
        f.write(f"{'Image':<15} {'Pos%':>7} {'Rot%':>7} {'Neigh%':>8} {'Time':>7}\n{'-'*46}\n")
        for r in results:
            f.write(f"{r['image']:<15} {r['pos_acc']:>6.1f}% {r['rot_acc']:>6.1f}% "
                    f"{r['neigh_acc']:>7.1f}% {r['time']:>6.1f}s\n")
        f.write(f"{'-'*46}\nAVERAGE         {avg_pos:>6.1f}% {avg_rot:>6.1f}% "
                f"{avg_neigh:>7.1f}% {total_t:>6.1f}s\n")


def parse_args():
    p = argparse.ArgumentParser(description="Jigsaw Puzzle Solver")
    p.add_argument("--image", type=str, help="Filename εικόνας (στο images_input/) ή full path")
    p.add_argument("--all",   action="store_true")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--ablation", action="store_true", help="Τρέχει ablation study μετά την επίλυση")
    p.add_argument("--clean-results", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.clean_results and os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
        print(f"Removed: {RESULTS_DIR}")

    if args.all:
        run_all(args.seed, run_abl=args.ablation)
    else:
        if not args.image:
            raise ValueError("Δώσε --image <filename> ή --all")
        path = args.image if os.path.isabs(args.image) else os.path.join(INPUT_DIR, args.image)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Δεν βρέθηκε: {path}")
        run_single(path, args.seed, run_abl=args.ablation)
