"""
puzzle_utils.py
Φόρτωση εικόνας, δυναμικό grid, center-crop, κόψιμο, ανακάτεμα, περιστροφή.
"""

import os
import pickle
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BORDER_WIDTH = 5
INPUT_DIR   = os.path.join(os.path.dirname(__file__), "images_input")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Pickle helpers ───────────────────────────────────────────────────
def save_pkl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def pkl_path(image_name, stage):
    return os.path.join(RESULTS_DIR, image_name, f"stage{stage}.pkl")


# ── Directory helpers ────────────────────────────────────────────────
def get_stage_paths(image_name):
    root = os.path.join(RESULTS_DIR, image_name)
    return {
        "root":   root,
        "stage1": os.path.join(root, "Stage_1_Shuffling"),
        "stage2": os.path.join(root, "Stage_2_Features"),
        "stage3": os.path.join(root, "Stage_3_Solving"),
    }


# ── Image helpers ────────────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Δεν βρέθηκε: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(tile, path):
    cv2.imwrite(path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

def save_grid(tiles, rows, cols, path, tile_size=None):
    th, tw = tile_size if tile_size else tiles[0].shape[:2]
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        t = tile if tile.shape[:2] == (th, tw) else cv2.resize(tile, (tw, th))
        canvas[r*th:(r+1)*th, c*tw:(c+1)*tw] = t
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ── Grid computation ─────────────────────────────────────────────────
def compute_dynamic_grid(h, w):
    rho = h / w
    if   rho >= 1.35: rows, cols = 5, 3
    elif rho >= 1.15: rows, cols = 4, 3
    elif rho <= 0.74: rows, cols = 3, 5
    elif rho <= 0.87: rows, cols = 3, 4
    else:             rows, cols = 3, 3
    # Clamp — tiles τουλάχιστον 96px, πλέγμα τουλάχιστον 3×3
    while rows > 3 and h // rows < 96: rows -= 1
    while cols > 3 and w // cols < 96: cols -= 1
    return rows, cols


# ── Tile operations ──────────────────────────────────────────────────
def center_crop(image, rows, cols):
    """Center-crop εικόνα ώστε να χωράει ακριβώς rows×cols τετράγωνα tiles."""
    h, w = image.shape[:2]
    s = min(h // rows, w // cols)          # κοινή πλευρά tile
    new_h, new_w = s * rows, s * cols
    top  = (h - new_h) // 2
    left = (w - new_w) // 2
    return image[top:top+new_h, left:left+new_w], s, s

def cut_tiles(image, rows, cols, th, tw):
    return [image[r*th:(r+1)*th, c*tw:(c+1)*tw].copy()
            for r in range(rows) for c in range(cols)]

def shuffle_and_rotate(tiles, seed=None):
    rng = np.random.default_rng(seed)
    n   = len(tiles)
    idx = rng.permutation(n)
    rot = rng.integers(0, 4, size=n)
    scrambled = [np.rot90(tiles[idx[i]], k=rot[i]).copy() for i in range(n)]
    return scrambled, idx, rot


# ── PuzzleBoard ──────────────────────────────────────────────────────
class PuzzleBoard:
    def __init__(self, image_path, seed=None):
        img = load_image(image_path)
        self.rows, self.cols = compute_dynamic_grid(*img.shape[:2])
        self.I_orig, self.tile_h, self.tile_w = center_crop(img, self.rows, self.cols)
        self.tiles = cut_tiles(self.I_orig, self.rows, self.cols, self.tile_h, self.tile_w)
        self.num_tiles = self.rows * self.cols
        _, self.shuffled_indices, self.applied_rotations = shuffle_and_rotate(self.tiles, seed)

    def get_shuffled_tiles(self):
        return [np.rot90(self.tiles[self.shuffled_indices[i]],
                         k=self.applied_rotations[i]).copy()
                for i in range(self.num_tiles)]


# ── Stage 1 runner ───────────────────────────────────────────────────
def run_stage1(image_path, seed=42, clear_prev=True, log_fn=print):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    paths = get_stage_paths(image_name)

    if clear_prev and os.path.isdir(paths["root"]):
        shutil.rmtree(paths["root"])
    os.makedirs(paths["stage1"], exist_ok=True)

    puzzle = PuzzleBoard(image_path, seed=seed)
    scrambled = puzzle.get_shuffled_tiles()

    save_img(puzzle.I_orig, os.path.join(paths["stage1"], "original_cropped.jpg"))
    save_grid(scrambled, puzzle.rows, puzzle.cols,
              os.path.join(paths["stage1"], "shuffled_puzzle.jpg"),
              tile_size=(puzzle.tile_h, puzzle.tile_w))

    save_pkl(pkl_path(image_name, 1), {'puzzle': puzzle, 'scrambled_tiles': scrambled})

    log_fn(f"[Stage 1] Φορτώνουμε {os.path.basename(image_path)} — "
           f"πλέγμα {puzzle.rows}×{puzzle.cols}, tile {puzzle.tile_h}×{puzzle.tile_w}px")

    return {'image_name': image_name, 'paths': paths,
            'puzzle': puzzle, 'scrambled_tiles': scrambled}


# ── Demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for f in sorted(os.listdir(INPUT_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            run_stage1(os.path.join(INPUT_DIR, f), seed=42)
