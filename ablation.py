"""
ablation.py — Ablation Study (§6.1)

Συγκρίνει 4 descriptor configurations:
  classical — pixels + color + texture + local (χωρίς deep)
  deep_only — μόνο deep features
  no_local  — combined χωρίς Harris+SIFT
  combined  — όλα μαζί (full pipeline)

Features υπολογίζονται ΜΙΑ ΦΟΡΑ (από pickle αν υπάρχει) — fair comparison.

Χρήση:
  python ablation.py --image egg.jpg --seed 42
"""

import os
import sys
import time
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from puzzle_utils import INPUT_DIR, RESULTS_DIR, get_stage_paths, run_stage1, \
                         save_grid, pkl_path, load_pkl
from features import run_stage2
from solver import JigsawSolver, ABLATION_CONFIGS


def _solve_config(tiles, puzzle, all_features, config_name, alpha_cfg, out_dir):
    """Greedy + local search για ένα ablation config. Επιστρέφει metrics dict."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    solver = JigsawSolver(puzzle.rows, puzzle.cols, alpha_cfg)
    solver.all_tile_features = all_features
    n = len(tiles)

    best_grid, best_s = None, -np.inf
    for s in range(n):
        for r in range(4):
            g  = solver._greedy_fill(s, r)
            sc = solver.score_grid(g)
            if sc > best_s:
                best_s = sc; best_grid = g.copy()

    best_grid = solver._local_search(best_grid)
    elapsed = time.time() - t0

    save_grid(solver.reconstruct(tiles, best_grid),
              puzzle.rows, puzzle.cols,
              os.path.join(out_dir, f"solved_{config_name}.jpg"),
              tile_size=(puzzle.tile_h, puzzle.tile_w))

    pos, rot, neigh = solver.evaluate(best_grid, puzzle.shuffled_indices, puzzle.applied_rotations)
    return {'config': config_name, 'pos_acc': pos, 'rot_acc': rot, 'neigh_acc': neigh, 'time': elapsed}


def _plot_comparison(results, path):
    configs = [r['config']    for r in results]
    pos_v   = [r['pos_acc']   for r in results]
    rot_v   = [r['rot_acc']   for r in results]
    neigh_v = [r['neigh_acc'] for r in results]
    x = np.arange(len(configs)); w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for vals, offset, label, clr in [
        (pos_v,   -w, 'Placement %', 'steelblue'),
        (rot_v,    0, 'Rotation %',  'coral'),
        (neigh_v,  w, 'Neighbor %',  'seagreen'),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label, color=clr, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + w/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 110)
    ax.set_title('Ablation Study — Descriptor Configuration Comparison (§6.1)')
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=10)
    ax.axhline(100, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3); ax.legend()

    best = int(np.argmax([r['pos_acc'] + r['rot_acc'] + r['neigh_acc'] for r in results]))
    ax.get_xticklabels()[best].set_fontweight('bold')
    ax.get_xticklabels()[best].set_color('darkblue')

    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def _print_table(results):
    sep = '─' * 60
    print(f"\n{sep}\nABLATION STUDY — Αποτελέσματα (§6.1)\n{sep}")
    print(f"{'Config':<12} {'Placement':>10} {'Rotation':>10} {'Neighbor':>10} {'Time':>7}")
    print(sep)
    for r in results:
        print(f"{r['config']:<12} {r['pos_acc']:>9.1f}% {r['rot_acc']:>9.1f}% "
              f"{r['neigh_acc']:>9.1f}% {r['time']:>6.1f}s")
    print(sep)
    scores = [r['pos_acc'] + r['rot_acc'] + r['neigh_acc'] for r in results]
    best = results[int(np.argmax(scores))]
    print(f"\n★ Καλύτερο: '{best['config']}' (combined score: {max(scores):.1f})")
    print("Ερμηνεία:")
    print("  classical vs combined → συνεισφορά deep features")
    print("  deep_only  vs combined → συνεισφορά classical descriptors")
    print("  no_local   vs combined → συνεισφορά Harris+SIFT")


def run_ablation(image_path, seed=42):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    abl_dir    = os.path.join(RESULTS_DIR, image_name, "Stage_4_Ablation")
    os.makedirs(abl_dir, exist_ok=True)

    print(f"\n{'═'*55}\nABLATION STUDY — {image_name}\n{'═'*55}")

    # Stage 1
    stage1  = run_stage1(image_path, seed=seed, clear_prev=False)
    puzzle  = stage1['puzzle']
    tiles   = stage1['scrambled_tiles']
    paths   = stage1['paths']

    # Stage 2 — φόρτωση από pickle αν υπάρχει, αλλιώς υπολογισμός
    pkl2 = pkl_path(image_name, 2)
    if os.path.exists(pkl2):
        print(f"[Stage 2] Loading cached features from pickle...")
        all_features = load_pkl(pkl2)
    else:
        os.makedirs(paths['stage2'], exist_ok=True)
        all_features = run_stage2(tiles, paths['stage2'])

    # Stage 4 — 4 configs
    print("\n[Stage 4] Running ablation configs...")
    results = []
    for cfg_name, alpha_cfg in ABLATION_CONFIGS.items():
        print(f"\n  ── {cfg_name}: active = {[k for k,v in alpha_cfg.items() if v > 0]}")
        r = _solve_config(tiles, puzzle, all_features, cfg_name, alpha_cfg,
                          os.path.join(abl_dir, cfg_name))
        results.append(r)

    _print_table(results)
    _plot_comparison(results, os.path.join(abl_dir, "ablation_comparison.png"))

    with open(os.path.join(abl_dir, "ablation_results.csv"), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['config', 'placement_%', 'rotation_%', 'neighbor_%', 'time_s'])
        for r in results:
            w.writerow([r['config'], f"{r['pos_acc']:.2f}", f"{r['rot_acc']:.2f}",
                        f"{r['neigh_acc']:.2f}", f"{r['time']:.1f}"])

    print(f"\nAblation results → {abl_dir}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--seed",  type=int, default=42)
    args = p.parse_args()
    path = args.image if os.path.isabs(args.image) else os.path.join(INPUT_DIR, args.image)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Δεν βρέθηκε: {path}")
    run_ablation(path, seed=args.seed)
