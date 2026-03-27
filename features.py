"""
features.py
Εξαγωγή χαρακτηριστικών (Descriptors) από τα πλακίδια του παζλ.

Descriptor families:
  Φ_col  — Color histogram per channel, L2-normalized (48 dim)
  Φ_tex  — Gabor filter bank + Sobel energy/mean/std (30 dim)
  Φ_loc  — Harris corners + SIFT-style gradient histogram (64 dim)
  Edge   — Resampled seam pixels σε σταθερό μέγεθος (192 dim)
  Φ_deep — ResNet-18 Global Average Pooling (512 dim)

Κάθε tile αναλύεται σε 4 περιστροφές × (tile-level + 4 side-level) features.
"""

import os
import time
import numpy as np
import cv2
import torch
from torchvision import models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Σταθερές ─────────────────────────────────────────────────────────
BINS        = 16     # bins/channel → 48 dim
BORDER_W    = 5      # πάχος border strip
EDGE_N      = 64     # samples ακμής → 192 dim
HARRIS_K    = 0.05
HARRIS_TOPK = 8      # keypoints ανά strip
SIFT_BINS   = 8      # orientation bins → 64 dim (8×8)
DEEP_SIZE   = 64     # resize strips για ResNet

GABOR_THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GABOR_WAVES  = [4, 8]
GABOR_SIGMA  = 3.0


# ── Border / edge helpers ─────────────────────────────────────────────
def get_strip(tile, side):
    h, w = tile.shape[:2]
    b = min(BORDER_W, h, w)
    s = {'top': (slice(0,b), slice(None)),
         'bottom': (slice(h-b,h), slice(None)),
         'left': (slice(None), slice(0,b)),
         'right': (slice(None), slice(w-b,w))}[side]
    return tile[s]

def get_edge_pixels(tile, side):
    """Γραμμή ραφής (1px) resampled σε EDGE_N δείγματα → 192 dim."""
    row = {'top': tile[0], 'bottom': tile[-1],
           'left': tile[:,0], 'right': tile[:,-1]}[side].astype(np.float64)
    n = row.shape[0]
    if n == EDGE_N:
        return row.flatten()
    xi = np.linspace(0, 1, EDGE_N)
    xo = np.linspace(0, 1, n)
    return np.stack([np.interp(xi, xo, row[:,ch]) for ch in range(3)], axis=1).flatten()


# ── Color histogram (Φ_col) ───────────────────────────────────────────
def color_hist(region):
    """L2-normalized per-channel histogram → 48 dim."""
    hists = []
    for ch in range(3):
        h = cv2.calcHist([region], [ch], None, [BINS], [0,256]).flatten().astype(np.float64)
        n = np.linalg.norm(h)
        hists.append(h / n if n > 0 else h)
    return np.concatenate(hists)


# ── Texture features (Φ_tex) ──────────────────────────────────────────
def _build_gabor_bank():
    ksize = int(6 * GABOR_SIGMA + 1) | 1
    return [cv2.getGaborKernel((ksize,ksize), GABOR_SIGMA, th, lam, 0.5, 0, cv2.CV_64F)
            for th in GABOR_THETAS for lam in GABOR_WAVES]

_gabor_bank = _build_gabor_bank()

def texture_features(region):
    """Gabor (8 filters) + Sobel (2) → mean/std/energy × 10 = 30 dim."""
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float64)
    resps = ([cv2.filter2D(gray, cv2.CV_64F, k) for k in _gabor_bank] +
             [cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
              cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)])
    return np.array([f(r) for r in resps for f in (np.mean, np.std, lambda x: np.mean(x**2))],
                    dtype=np.float64)


# ── Local interest points (Φ_loc): Harris + SIFT ─────────────────────
def _harris_response(gray):
    gf = gray.astype(np.float32)
    Ix = cv2.Sobel(gf, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gf, cv2.CV_64F, 0, 1, ksize=3)
    Ix2  = cv2.GaussianBlur(Ix*Ix,  (5,5), 1.5)
    Iy2  = cv2.GaussianBlur(Iy*Iy,  (5,5), 1.5)
    IxIy = cv2.GaussianBlur(Ix*Iy, (5,5), 1.5)
    return Ix2*Iy2 - IxIy**2 - HARRIS_K*(Ix2+Iy2)**2, Ix, Iy

def _top_keypoints(R, k=HARRIS_TOPK):
    h, w = R.shape
    pts = np.argwhere(R > 0)
    if len(pts) == 0:
        return list(zip(np.full(k, h//2), np.linspace(1, w-2, k, dtype=int)))
    order = np.argsort(-R[pts[:,0], pts[:,1]])
    kps = []
    for i in order:
        y, x = pts[i]
        if all(abs(y-ky) >= 3 or abs(x-kx) >= 3 for ky, kx in kps):
            kps.append((int(y), int(x)))
        if len(kps) >= k:
            break
    while len(kps) < k:
        kps.append((h//2, len(kps) * max(1, w//k) % max(1, w-1)))
    return kps

def _sift_desc(gray, Ix, Iy, y, x, r=4):
    h, w = gray.shape
    y0, y1 = max(0, y-r), min(h, y+r+1)
    x0, x1 = max(0, x-r), min(w, x+r+1)
    mag = np.sqrt(Ix[y0:y1,x0:x1]**2 + Iy[y0:y1,x0:x1]**2)
    ori = np.arctan2(Iy[y0:y1,x0:x1], Ix[y0:y1,x0:x1]) % (2*np.pi)
    py, px = np.mgrid[y0:y1, x0:x1]
    wm = mag * np.exp(-((py-y)**2 + (px-x)**2) / (2*max(1.0, r)**2))
    bw = 2*np.pi / SIFT_BINS
    desc = np.array([
        np.sum(wm * np.exp(-((ori-(b+.5)*bw+np.pi) % (2*np.pi) - np.pi)**2 / (2*bw**2)))
        for b in range(SIFT_BINS)
    ])
    n = np.linalg.norm(desc)
    return desc / n if n > 1e-8 else desc

def local_descriptor(strip):
    """Harris + SIFT → HARRIS_TOPK × SIFT_BINS = 64 dim, L2-norm."""
    h, w = strip.shape[:2]
    if h < 16 or w < 16:
        scale = max(16/h, 16/w)
        strip = cv2.resize(strip, (max(16, int(w*scale)), max(16, int(h*scale))))
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY).astype(np.float64)
    R, Ix, Iy = _harris_response(gray)
    desc = np.concatenate([_sift_desc(gray, Ix, Iy, y, x) for y, x in _top_keypoints(R)])
    n = np.linalg.norm(desc)
    return desc / n if n > 1e-8 else desc


# ── Deep CNN (Φ_deep) ─────────────────────────────────────────────────
class _DeepExtractor:
    """ResNet-18 χωρίς τελευταίο FC → Global Average Pooling → 512 dim."""
    def __init__(self):
        w = models.ResNet18_Weights.DEFAULT
        m = models.resnet18(weights=w)
        self.net = torch.nn.Sequential(*list(m.children())[:-1])
        self.net.eval()
        self.pre = w.transforms()

    def __call__(self, region):
        t = self.pre(Image.fromarray(region)).unsqueeze(0)
        with torch.no_grad():
            return self.net(t).flatten().numpy().astype(np.float64)

    def from_strip(self, strip):
        return self(cv2.resize(strip, (DEEP_SIZE, DEEP_SIZE)))

_deep = None
def _get_deep():
    global _deep
    if _deep is None:
        _deep = _DeepExtractor()
    return _deep


# ── Κεντρική συνάρτηση εξαγωγής ──────────────────────────────────────
def extract_all_features(tile):
    """
    Εξάγει ΟΛΑ τα features για ένα tile.
    Επιστρέφει dict με side-level (×4 πλευρές) και tile-level descriptors.
    """
    deep = _get_deep()
    f = {
        'deep':      deep(tile),
        'tile_hist': color_hist(tile),
        'tile_tex':  texture_features(tile),
    }
    for side in ('top', 'bottom', 'left', 'right'):
        strip = get_strip(tile, side)
        f[f'{side}_hist']   = color_hist(strip)
        f[f'{side}_tex']    = texture_features(strip)
        f[f'{side}_pixels'] = get_edge_pixels(tile, side)
        f[f'{side}_local']  = local_descriptor(strip)
        f[f'{side}_deep']   = deep.from_strip(strip)
    return f


# ── Stage 2 runner ────────────────────────────────────────────────────
def run_stage2(scrambled_tiles, stage2_dir, log_fn=print, progress_fn=None):
    """
    Εξάγει features για όλα τα tiles (× 4 rotations).
    Αποθηκεύει: avg_tile_histogram.png + stage2.pkl
    Επιστρέφει: all_features[i][rot] = feature_dict
    """
    from puzzle_utils import save_pkl, pkl_path
    os.makedirs(stage2_dir, exist_ok=True)

    n = len(scrambled_tiles)
    t0 = time.time()
    all_features = {}

    log_fn(f"[Stage 2] Τρέχουμε features ({n} tiles × 4 rotations)...")
    for i, tile in enumerate(scrambled_tiles):
        all_features[i] = {rot: extract_all_features(np.rot90(tile, k=rot))
                           for rot in range(4)}
        # CLI progress bar
        done = i + 1
        bar_len = 20
        filled = int(bar_len * done / n)
        bar = '█' * filled + ' ' * (bar_len - filled)
        print(f"\r  [{bar}] {done}/{n} tiles", end='', flush=True)
        if progress_fn:
            progress_fn("Stage 2", int(done / n * 100))
    print()

    # Avg tile color histogram — μόνο αυτή η οπτικοποίηση
    avg_hist = np.mean([all_features[i][0]['tile_hist'] for i in range(n)], axis=0)
    _save_avg_histogram(avg_hist, os.path.join(stage2_dir, "avg_tile_histogram.png"))

    # Pickle για χρήση από επόμενα στάδια / ablation
    image_name = os.path.basename(os.path.dirname(stage2_dir))
    save_pkl(pkl_path(image_name, 2), all_features)

    log_fn(f"[Stage 2] Ολοκληρώθηκε σε {time.time()-t0:.1f}s")
    return all_features


def _save_avg_histogram(hist, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ch, (ax, clr, lbl) in enumerate(zip(axes, ['red','green','blue'], ['R','G','B'])):
        ax.bar(range(BINS), hist[ch*BINS:(ch+1)*BINS], color=clr, alpha=0.7)
        ax.set_title(f"{lbl} Channel"); ax.set_xlabel("Bin"); ax.set_ylabel("L2-Norm Value")
    fig.suptitle("Average Tile Histogram (mean across tiles)", fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=100, bbox_inches='tight'); plt.close(fig)


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from puzzle_utils import PuzzleBoard, get_stage_paths, INPUT_DIR
    p = PuzzleBoard(os.path.join(INPUT_DIR, "Io.jpg"), seed=42)
    feats = run_stage2(p.get_shuffled_tiles(), get_stage_paths("Io")["stage2"])
    f0 = feats[0][0]
    print({k: v.shape for k, v in f0.items()})
