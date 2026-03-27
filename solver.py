"""
solver.py
Greedy Solver + Local Search για Jigsaw puzzle.

Compatibility score (§5.4):
  C((i,s),(j,t)) = Σ_d α_d [ w_side·g(d_side) + w_tile·g(d_tile) ]
  g(d) = exp(-λ · mean((a-b)²))

Solver pipeline:
  1. Exhaustive greedy (N×4 starting points)
  2. Local search (swap + rotate)
  3. Rotation-aware evaluation (§6.1)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Βάρη ─────────────────────────────────────────────────────────────
ALPHA  = {'pixels': 1.0, 'color': 0.5, 'texture': 0.2, 'local': 0.4, 'deep': 0.3}
W_SIDE = {'pixels': 1.0, 'color': 0.7, 'texture': 0.7, 'local': 1.0, 'deep': 0.4}
W_TILE = {'pixels': 0.0, 'color': 0.3, 'texture': 0.3, 'local': 0.0, 'deep': 0.6}
LAMBDA = {'pixels': 1e-4, 'color': 5.0, 'texture': 0.01, 'local': 10.0, 'deep': 2.0}

# descriptor → key suffix στο feature dict
SIDE_KEY = {'pixels': 'pixels', 'color': 'hist', 'texture': 'tex',
            'local': 'local', 'deep': 'deep'}
TILE_KEY = {'color': 'tile_hist', 'texture': 'tile_tex', 'deep': 'deep'}

ABLATION_CONFIGS = {
    'classical': {**ALPHA, 'deep': 0.0},
    'deep_only': {'pixels': 0.0, 'color': 0.0, 'texture': 0.0, 'local': 0.0, 'deep': 1.0},
    'no_local':  {**ALPHA, 'local': 0.0},
    'combined':  ALPHA,
}

MAX_LOCAL_ITERS = 200


def _gsim(a, b, lam):
    """Gaussian similarity: exp(-λ · mean((a-b)²))"""
    return np.exp(-lam * np.mean((a - b) ** 2))


class JigsawSolver:
    def __init__(self, rows, cols, alpha_config=None):
        self.rows = rows
        self.cols = cols
        self.N    = rows * cols
        self._alpha = alpha_config or ALPHA
        self.all_tile_features = {}

    # ── Compatibility score ───────────────────────────────────────────
    def get_compatibility(self, fa, fb, sa, sb):
        """
        Σ_d α_d [w_side·g_side + w_tile·g_tile]
        Όλα τα descriptors: _gsim (mean SSD + Gaussian).
        Deep tile-level: cosine similarity (semantic embedding).
        """
        total = 0.0
        for d, alpha in self._alpha.items():
            if not alpha:
                continue
            lam, ws, wt = LAMBDA[d], W_SIDE[d], W_TILE[d]
            c = 0.0
            if ws:
                c += ws * _gsim(fa[f'{sa}_{SIDE_KEY[d]}'], fb[f'{sb}_{SIDE_KEY[d]}'], lam)
            if wt and d in TILE_KEY:
                if d == 'deep':
                    da, db = fa['deep'], fb['deep']
                    cos = np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12)
                    c += wt * (cos + 1) / 2
                else:
                    c += wt * _gsim(fa[TILE_KEY[d]], fb[TILE_KEY[d]], lam)
            total += alpha * c
        return total

    # ── F_adj για ολόκληρο grid ───────────────────────────────────────
    def score_grid(self, grid):
        total = 0.0
        feats = self.all_tile_features
        for r in range(self.rows):
            for c in range(self.cols):
                i, ri = grid[r, c]
                fa = feats[i][ri]
                if c + 1 < self.cols:
                    j, rj = grid[r, c+1]
                    total += self.get_compatibility(fa, feats[j][rj], 'right', 'left')
                if r + 1 < self.rows:
                    j, rj = grid[r+1, c]
                    total += self.get_compatibility(fa, feats[j][rj], 'bottom', 'top')
        return total

    # ── Greedy fill ───────────────────────────────────────────────────
    def _greedy_fill(self, start_idx, start_rot):
        """Row-by-row greedy: τοποθετεί tile με max avg compatibility με τους ήδη τοποθετημένους γείτονες."""
        grid = np.full((self.rows, self.cols, 2), -1, dtype=int)
        grid[0, 0] = [start_idx, start_rot]
        used = {start_idx}
        feats = self.all_tile_features
        for r in range(self.rows):
            for c in range(self.cols):
                if r == 0 and c == 0:
                    continue
                best, best_s = (-1, -1), -np.inf
                for t in range(self.N):
                    if t in used:
                        continue
                    for rot in range(4):
                        f = feats[t][rot]
                        s, n = 0.0, 0
                        if c > 0:
                            l, lr = grid[r, c-1]
                            s += self.get_compatibility(feats[l][lr], f, 'right', 'left')
                            n += 1
                        if r > 0:
                            u, ur = grid[r-1, c]
                            s += self.get_compatibility(feats[u][ur], f, 'bottom', 'top')
                            n += 1
                        avg = s / n if n else s
                        if avg > best_s:
                            best_s = avg; best = (t, rot)
                grid[r, c] = best; used.add(best[0])
        return grid

    # ── Local search ──────────────────────────────────────────────────
    def _local_search(self, grid):
        best = grid.copy()
        best_s = self.score_grid(best)
        improved, it = True, 0
        while improved and it < MAX_LOCAL_ITERS:
            improved = False; it += 1
            pos = [(r, c) for r in range(self.rows) for c in range(self.cols)]
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    trial = best.copy()
                    r1,c1 = pos[i]; r2,c2 = pos[j]
                    trial[r1,c1], trial[r2,c2] = trial[r2,c2].copy(), trial[r1,c1].copy()
                    s = self.score_grid(trial)
                    if s > best_s:
                        best = trial; best_s = s; improved = True
            for r in range(self.rows):
                for c in range(self.cols):
                    for new_rot in range(4):
                        if new_rot == best[r,c,1]:
                            continue
                        trial = best.copy(); trial[r,c,1] = new_rot
                        s = self.score_grid(trial)
                        if s > best_s:
                            best = trial; best_s = s; improved = True
        print(f"  Τοπική βελτιστοποίηση: {it} επανάληψη/εις, F_adj = {best_s:.4f}")
        return best

    def reconstruct(self, tiles, grid):
        return [np.rot90(tiles[grid[r,c,0]], k=grid[r,c,1])
                for r in range(self.rows) for c in range(self.cols)]

    # ── Evaluation (§6.1) rotation-aware ─────────────────────────────
    def evaluate(self, grid, shuffled_indices, applied_rotations):
        """Δοκιμάζει 4 global rotations, κρατάει το καλύτερο αποτέλεσμα."""
        best, best_s = None, -1
        for g_rot in range(4):
            if g_rot in (1, 3) and self.rows != self.cols:
                continue
            res = self._eval_rot(grid, shuffled_indices, applied_rotations, g_rot)
            if res and sum(res[:3]) > best_s:
                best_s = sum(res[:3]); best = res
        pos, rot, neigh, g_rot = best
        if g_rot:
            print(f"  Ολική περιστροφή: {g_rot*90}°")
        print(f"  Θέση: {pos:.1f}%  |  Περιστροφή: {rot:.1f}%  |  Γειτνίαση: {neigh:.1f}%")
        return pos, rot, neigh

    def _eval_rot(self, grid, s_idx, s_rot, g_rot):
        rg = np.rot90(grid, k=g_rot, axes=(0,1)).copy()
        if rg.shape[:2] != (self.rows, self.cols):
            return None
        pos_ok = rot_ok = 0
        total = self.N
        sol = {}
        for r in range(self.rows):
            for c in range(self.cols):
                pi, pr = int(rg[r,c,0]), int(rg[r,c,1])
                real = s_idx[pi]; sol[(r,c)] = real
                if real == r * self.cols + c:
                    pos_ok += 1
                    if (s_rot[pi] + pr + g_rot) % 4 == 0:
                        rot_ok += 1
        neigh_ok = neigh_tot = 0
        for r in range(self.rows):
            for c in range(self.cols):
                for dr, dc in ((0,1),(1,0)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        neigh_tot += 1
                        if sol.get((r,c)) == r*self.cols+c and sol.get((nr,nc)) == nr*self.cols+nc:
                            neigh_ok += 1
        return pos_ok/total*100, rot_ok/total*100, (neigh_ok/neigh_tot*100 if neigh_tot else 0), g_rot


# ── Compatibility heatmap ─────────────────────────────────────────────
def save_compat_heatmap(solver, path):
    n = solver.N
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                fi = solver.all_tile_features[i][0]
                C[i,j] = max(solver.get_compatibility(fi, solver.all_tile_features[j][rot], 'right', 'left')
                             for rot in range(4))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C, cmap='viridis')
    fig.colorbar(im, ax=ax, label='Compatibility Score')
    ax.set_title("Compatibility Matrix (best rot, right→left)")
    ax.set_xlabel("Tile j"); ax.set_ylabel("Tile i")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if C[i,j] < C.max()*0.5 else 'black')
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)


# ── Stage 3 runner ────────────────────────────────────────────────────
def run_stage3(scrambled_tiles, puzzle, all_features, stage3_dir,
               log_fn=print, progress_fn=None, alpha_config=None):
    from puzzle_utils import save_grid
    os.makedirs(stage3_dir, exist_ok=True)

    solver = JigsawSolver(puzzle.rows, puzzle.cols, alpha_config)
    solver.all_tile_features = all_features
    n = len(scrambled_tiles)
    t0 = time.time()

    log_fn(f"[Stage 3] Greedy search ({n} tiles × 4 starts)...")
    best_grid, best_s = None, -np.inf
    total_tries = n * 4
    for s in range(n):
        for r in range(4):
            g = solver._greedy_fill(s, r)
            sc = solver.score_grid(g)
            if sc > best_s:
                best_s = sc; best_grid = g.copy()
            done = s * 4 + r + 1
            bar_len = 20
            filled = int(bar_len * done / total_tries)
            bar = '█' * filled + ' ' * (bar_len - filled)
            print(f"\r  [{bar}] {done}/{total_tries} starts", end='', flush=True)
    print()
    log_fn(f"  Καλύτερο greedy F_adj = {best_s:.4f}")
    if progress_fn: progress_fn("Stage 3", 50)

    log_fn(f"  Τοπική βελτιστοποίηση...")
    best_grid = solver._local_search(best_grid)
    elapsed = time.time() - t0

    save_grid(solver.reconstruct(scrambled_tiles, best_grid),
              puzzle.rows, puzzle.cols,
              os.path.join(stage3_dir, "solved_puzzle.jpg"),
              tile_size=(puzzle.tile_h, puzzle.tile_w))
    log_fn(f"  Αποθήκευση compatibility heatmap...")
    save_compat_heatmap(solver, os.path.join(stage3_dir, "compatibility_matrix.png"))

    pos, rot, neigh = solver.evaluate(best_grid, puzzle.shuffled_indices, puzzle.applied_rotations)
    log_fn(f"  Χρόνος επίλυσης: {elapsed:.1f}s")

    image_name = os.path.basename(os.path.dirname(stage3_dir))
    with open(os.path.join(stage3_dir, "evaluation.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results — {image_name}\n{'='*45}\n")
        f.write(f"Grid: {puzzle.rows}×{puzzle.cols}  |  Tile: {puzzle.tile_h}×{puzzle.tile_w}  |  Time: {elapsed:.1f}s\n\n")
        f.write(f"Piece Placement Accuracy: {pos:.1f}%\n")
        f.write(f"Rotation Accuracy:        {rot:.1f}%\n")
        f.write(f"Neighbor Accuracy:        {neigh:.1f}%\n")

    if progress_fn: progress_fn("Stage 3", 100)
    return {'pos_acc': pos, 'rot_acc': rot, 'neigh_acc': neigh, 'time': elapsed, 'best_grid': best_grid}


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from puzzle_utils import PuzzleBoard, get_stage_paths, INPUT_DIR
    from features import run_stage2
    p = PuzzleBoard(os.path.join(INPUT_DIR, "Io.jpg"), seed=42)
    sc = p.get_shuffled_tiles()
    paths = get_stage_paths("Io")
    feats = run_stage2(sc, paths["stage2"])
    run_stage3(sc, p, feats, paths["stage3"])
