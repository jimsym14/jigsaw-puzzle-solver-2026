"""
gui.py — Απλό interactive interface για το Jigsaw Puzzle Solver.

Χρησιμοποιεί tkinter + matplotlib.
Τρέχει με: python gui.py

Απαιτήσεις: pip install matplotlib pillow
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Lazy imports από τον κώδικά μας ─────────────────────────────────
def _import_project():
    from puzzle_utils import PuzzleBoard, run_stage1, INPUT_DIR, RESULTS_DIR, get_stage_paths
    from features     import run_stage2
    from solver       import run_stage3, ABLATION_CONFIGS
    return PuzzleBoard, run_stage1, run_stage2, run_stage3, ABLATION_CONFIGS, INPUT_DIR, RESULTS_DIR, get_stage_paths


# ════════════════════════════════════════════════════════════════════
#  ΚΕΝΤΡΙΚΟ ΠΑΡΑΘΥΡΟ
# ════════════════════════════════════════════════════════════════════
class JigsawGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Jigsaw Puzzle Solver")
        self.geometry("1100x750")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        # state
        self.image_path   = tk.StringVar()
        self.seed_var     = tk.StringVar(value="42")
        self.current_stage = 0
        self._puzzle      = None
        self._scrambled   = None
        self._all_features = None
        self._stage3_result = None

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────
    #  UI LAYOUT
    # ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Αριστερά: controls + log ──────────────────────────────
        left = tk.Frame(self, bg="#1e1e2e", width=260)
        left.pack(side="left", fill="y", padx=12, pady=12)
        left.pack_propagate(False)

        tk.Label(left, text="Jigsaw Puzzle Solver",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(4,14))

        # Image path + seed
        tk.Label(left, text="Εικόνα:", bg="#1e1e2e", fg="#a6adc8",
             font=("Helvetica", 10)).pack(anchor="w")
        row_img = tk.Frame(left, bg="#1e1e2e")
        row_img.pack(fill="x", pady=(2,2))
        tk.Entry(row_img, textvariable=self.image_path,
             bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
             relief="flat", font=("Courier", 9)).pack(side="left", fill="x", expand=True)
        tk.Button(row_img, text="…", command=self._browse,
              bg="#45475a", fg="#cdd6f4", relief="flat",
              padx=6).pack(side="left", padx=(4,0))

        row_seed = tk.Frame(left, bg="#1e1e2e")
        row_seed.pack(fill="x", pady=(0,10))
        tk.Label(row_seed, text="Seed:", bg="#1e1e2e", fg="#a6adc8",
             font=("Helvetica", 10)).pack(side="left")
        tk.Entry(row_seed, textvariable=self.seed_var, width=7,
             bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
             relief="flat", font=("Courier", 9)).pack(side="left", padx=(6,0))

        # Stage buttons
        self._btns = {}
        stages = [
            ("stage1", "▶  Stage 1 — Shuffle",   self._run_stage1),
            ("stage2", "▶  Stage 2 — Features",  self._run_stage2),
            ("stage3", "▶  Stage 3 — Solve",     self._run_stage3),
            ("ablation","▶  Ablation Study",      self._run_ablation),
        ]
        for key, label, cmd in stages:
            b = tk.Button(left, text=label, command=cmd,
                          bg="#313244", fg="#cdd6f4", relief="flat",
                          font=("Helvetica", 10), pady=7, anchor="w", padx=10,
                          state="disabled" if key != "stage1" else "normal",
                          disabledforeground="#585b70")
            b.pack(fill="x", pady=3)
            self._btns[key] = b

        # Progress bar
        tk.Label(left, text="Πρόοδος:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Helvetica", 10)).pack(anchor="w", pady=(14,2))
        self._progress = ttk.Progressbar(left, maximum=100)
        self._progress.pack(fill="x")
        self._prog_label = tk.Label(left, text="", bg="#1e1e2e", fg="#6c7086",
                                     font=("Courier", 9))
        self._prog_label.pack(anchor="w")

        # Log
        tk.Label(left, text="Log:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Helvetica", 10)).pack(anchor="w", pady=(14,2))
        self._log = scrolledtext.ScrolledText(
            left, height=14, bg="#181825", fg="#a6e3a1",
            font=("Courier", 8), relief="flat", wrap="word",
            insertbackground="#cdd6f4")
        self._log.pack(fill="both", expand=True)

        # ── Δεξιά: matplotlib canvas ──────────────────────────────
        right = tk.Frame(self, bg="#1e1e2e")
        right.pack(side="left", fill="both", expand=True, padx=(0,12), pady=12)

        self._fig = plt.Figure(figsize=(8.5, 6.5), facecolor="#1e1e2e")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self._show_welcome()

    # ─────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if path:
            self.image_path.set(path)

    def _log_msg(self, msg):
        self._log.insert("end", msg + "\n")
        self._log.see("end")

    def _set_progress(self, stage_label, pct):
        self._progress["value"] = pct
        self._prog_label.config(text=f"{stage_label}: {pct}%")
        self.update_idletasks()

    def _enable_btn(self, key):
        self._btns[key].config(state="normal", bg="#45475a")

    def _draw(self):
        self._canvas.draw()

    # ─────────────────────────────────────────────────────────────────
    #  WELCOME SCREEN
    # ─────────────────────────────────────────────────────────────────
    def _show_welcome(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor("#1e1e2e")
        ax.text(0.5, 0.6, "Jigsaw Puzzle Solver",
                ha="center", va="center", fontsize=22,
                color="#cdd6f4", fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.45,
                "1. Επίλεξε εικόνα  →  2. Stage 1  →  3. Stage 2  →  4. Stage 3",
                ha="center", va="center", fontsize=11,
                color="#a6adc8", transform=ax.transAxes)
        ax.axis("off")
        self._draw()

    # ─────────────────────────────────────────────────────────────────
    #  STAGE 1
    # ─────────────────────────────────────────────────────────────────
    def _run_stage1(self):
        path = self.image_path.get().strip()
        if not path or not os.path.isfile(path):
            self._log_msg("✗ Δεν βρέθηκε εικόνα. Επέλεξε αρχείο πρώτα.")
            return

        self._log_msg("── Stage 1: Shuffle & Rotate ──")
        self._set_progress("Stage 1", 10)

        try:
            _, run_stage1_fn, _, _, _, INPUT_DIR, RESULTS_DIR, get_stage_paths = _import_project()
        except Exception as e:
            self._log_msg(f"✗ Import error: {e}")
            return

        try:
            # Πάρε το seed από το input
            try:
                seed = int(self.seed_var.get())
            except Exception:
                seed = 42
                self.seed_var.set("42")
                self._log_msg("✗ Μη έγκυρο seed, χρησιμοποιείται το 42.")
            result = run_stage1_fn(path, seed=seed,
                                   clear_prev=True,
                                   log_fn=self._log_msg)
            self._puzzle    = result["puzzle"]
            self._scrambled = result["scrambled_tiles"]
            self._set_progress("Stage 1", 100)
            self._log_msg(f"✓ Grid {self._puzzle.rows}×{self._puzzle.cols} | "
                          f"tile {self._puzzle.tile_h}×{self._puzzle.tile_w}")
            self._plot_stage1()
            self._enable_btn("stage2")
        except Exception as e:
            self._log_msg(f"✗ {e}")

    def _plot_stage1(self):
        p  = self._puzzle
        sc = self._scrambled
        rows, cols = p.rows, p.cols
        n = rows * cols

        self._fig.clear()
        self._fig.suptitle("Stage 1 — Original (πάνω)  vs  Shuffled & Rotated (κάτω)",
                            color="#cdd6f4", fontsize=12, y=0.98)

        rot_labels = {0:"0°", 1:"90°", 2:"180°", 3:"270°"}

        gs = gridspec.GridSpec(2*rows, cols, figure=self._fig,
                               hspace=0.15, wspace=0.05,
                               left=0.02, right=0.98, top=0.92, bottom=0.04)

        # Πάνω: original (στη σωστή θέση)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                ax = self._fig.add_subplot(gs[r, c])
                ax.imshow(p.tiles[idx])
                ax.set_title(f"({r},{c})", fontsize=7, color="#a6adc8", pad=2)
                ax.axis("off")
                _frame(ax, "#89b4fa")

        # Κάτω: shuffled (στη θέση που βρέθηκε μετά το shuffle)
        # Πρέπει να βρούμε για κάθε θέση (r, c) ποιο tile κατέληξε εκεί
        # Δηλαδή, για κάθε θέση i = r*cols+c, βρίσκουμε το index του tile που πήγε εκεί
        # Τα sc είναι ήδη στη σωστή σειρά (δηλ. sc[i] είναι το tile που πήγε στη θέση i)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                ax = self._fig.add_subplot(gs[rows + r, c])
                ax.imshow(sc[idx])
                # Ποιο tile ήταν αυτό πριν το shuffle;
                orig_idx = p.shuffled_indices[idx] if hasattr(p, 'shuffled_indices') else idx
                rot = p.applied_rotations[idx] if hasattr(p, 'applied_rotations') else 0
                rot_deg = rot_labels.get(rot, "?")
                ax.set_title(f"rot {rot_deg}", fontsize=7, color="#f38ba8", pad=2)
                ax.axis("off")
                _frame(ax, "#f38ba8")

        # row labels
        self._fig.text(0.005, 0.5 + 0.45/(2*rows), "Original", color="#89b4fa",
                       fontsize=9, va="center", rotation=90)
        self._fig.text(0.005, 0.5 - 0.45/(2*rows), "Shuffled", color="#f38ba8",
                       fontsize=9, va="center", rotation=90)
        self._draw()

    # ─────────────────────────────────────────────────────────────────
    #  STAGE 2
    # ─────────────────────────────────────────────────────────────────
    def _run_stage2(self):
        if self._puzzle is None:
            self._log_msg("✗ Τρέξε πρώτα Stage 1."); return

        self._log_msg("── Stage 2: Feature Extraction ──")
        self._btns["stage2"].config(state="disabled")

        _, _, run_stage2_fn, _, _, _, RESULTS_DIR, get_stage_paths = _import_project()

        image_name = os.path.splitext(os.path.basename(self.image_path.get()))[0]
        stage2_dir = get_stage_paths(image_name)["stage2"]
        os.makedirs(stage2_dir, exist_ok=True)

        def _worker():
            try:
                feats = run_stage2_fn(
                    self._scrambled, stage2_dir,
                    log_fn=self._log_msg,
                    progress_fn=self._set_progress,
                )
                self._all_features = feats
                self.after(0, self._plot_stage2)
                self.after(0, lambda: self._enable_btn("stage3"))
                self.after(0, lambda: self._log_msg("✓ Features extracted."))
            except Exception as e:
                self.after(0, lambda: self._log_msg(f"✗ {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _plot_stage2(self):
        feats_tile0_rot0 = self._all_features[0][0]
        sides = ["top", "right", "bottom", "left"]

        self._fig.clear()
        self._fig.suptitle("Stage 2 — Features για Tile 0 (rot=0°)",
                            color="#cdd6f4", fontsize=12, y=0.98)

        gs = gridspec.GridSpec(3, 4, figure=self._fig,
                               hspace=0.55, wspace=0.40,
                               left=0.06, right=0.97, top=0.90, bottom=0.06)

        BINS = 16
        colors_rgb = ["#f38ba8", "#a6e3a1", "#89b4fa"]

        # ── Row 0: Color histograms ανά side ─────────────────────
        for si, side in enumerate(sides):
            ax = self._fig.add_subplot(gs[0, si])
            h = feats_tile0_rot0[f"{side}_hist"]   # 48-dim
            for ch in range(3):
                ax.bar(range(ch, BINS*3, 3), h[ch*BINS:(ch+1)*BINS],
                       color=colors_rgb[ch], alpha=0.7, width=1)
            ax.set_title(f"Φ_col — {side}", fontsize=8, color="#a6adc8")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#181825")
            for sp in ax.spines.values(): sp.set_color("#313244")

        # ── Row 1: Texture (Gabor+Sobel) ──────────────────────────
        for si, side in enumerate(sides):
            ax = self._fig.add_subplot(gs[1, si])
            t = feats_tile0_rot0[f"{side}_tex"]    # 30-dim
            n_f = 10
            means = t[:n_f]; stds = t[n_f:2*n_f]; ens = t[2*n_f:]
            x = np.arange(n_f)
            ax.bar(x - 0.25, np.abs(means), 0.25, color="#89b4fa", alpha=0.8, label="μ")
            ax.bar(x,        stds,           0.25, color="#a6e3a1", alpha=0.8, label="σ")
            ax.bar(x + 0.25, ens,            0.25, color="#f9e2af", alpha=0.8, label="E")
            ax.set_title(f"Φ_tex — {side}", fontsize=8, color="#a6adc8")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#181825")
            for sp in ax.spines.values(): sp.set_color("#313244")
            if si == 0:
                ax.legend(fontsize=6, loc="upper right",
                          facecolor="#313244", labelcolor="#cdd6f4", framealpha=0.8)

        # ── Row 2: Edge pixels + Local descriptor ─────────────────
        # Left 2: edge pixels για top & right
        for si, side in enumerate(["top", "right"]):
            ax = self._fig.add_subplot(gs[2, si])
            px = feats_tile0_rot0[f"{side}_pixels"]  # 192-dim
            n_s = len(px) // 3
            rgb = px.reshape(n_s, 3)
            ax.plot(rgb[:, 0], color="#f38ba8", lw=0.8, label="R")
            ax.plot(rgb[:, 1], color="#a6e3a1", lw=0.8, label="G")
            ax.plot(rgb[:, 2], color="#89b4fa", lw=0.8, label="B")
            ax.set_title(f"Edge pixels — {side}", fontsize=8, color="#a6adc8")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#181825")
            for sp in ax.spines.values(): sp.set_color("#313244")
            if si == 0:
                ax.legend(fontsize=6, facecolor="#313244",
                          labelcolor="#cdd6f4", framealpha=0.8)

        # Right 2: local descriptor (Harris+SIFT) — αν υπάρχει
        for si, side in enumerate(["top", "right"]):
            ax = self._fig.add_subplot(gs[2, si + 2])
            key = f"{side}_local"
            if key in feats_tile0_rot0:
                loc = feats_tile0_rot0[key]  # 64-dim
                n_kp, n_bins = 8, 8
                data = loc.reshape(n_kp, n_bins)
                im = ax.imshow(data, aspect="auto", cmap="viridis",
                               interpolation="nearest")
                ax.set_title(f"Φ_loc Harris+SIFT — {side}", fontsize=8, color="#a6adc8")
                ax.set_xlabel("Ori bins", fontsize=6, color="#6c7086")
                ax.set_ylabel("Keypoints", fontsize=6, color="#6c7086")
                ax.tick_params(labelsize=6, colors="#6c7086")
            else:
                ax.text(0.5, 0.5, "Φ_loc\n(Harris+SIFT)\n— not available —",
                        ha="center", va="center", fontsize=8, color="#6c7086",
                        transform=ax.transAxes)
                ax.set_title(f"Φ_loc — {side}", fontsize=8, color="#a6adc8")
                ax.axis("off")
            ax.set_facecolor("#181825")
            for sp in ax.spines.values(): sp.set_color("#313244")

        self._draw()

    # ─────────────────────────────────────────────────────────────────
    #  STAGE 3
    # ─────────────────────────────────────────────────────────────────
    def _run_stage3(self):
        if self._all_features is None:
            self._log_msg("✗ Τρέξε πρώτα Stage 2."); return

        self._log_msg("── Stage 3: Greedy + Local Search ──")
        self._btns["stage3"].config(state="disabled")

        _, _, _, run_stage3_fn, _, _, RESULTS_DIR, get_stage_paths = _import_project()

        image_name = os.path.splitext(os.path.basename(self.image_path.get()))[0]
        stage3_dir = get_stage_paths(image_name)["stage3"]
        os.makedirs(stage3_dir, exist_ok=True)

        def _worker():
            try:
                res = run_stage3_fn(
                    self._scrambled,
                    self._puzzle,
                    self._all_features,
                    stage3_dir,
                    log_fn=self._log_msg,
                    progress_fn=self._set_progress,
                )
                self._stage3_result = res
                self.after(0, self._plot_stage3)
                self.after(0, lambda: self._enable_btn("ablation"))
                self.after(0, lambda: self._log_msg(
                    f"✓ Placement={res['pos_acc']:.1f}%  "
                    f"Rotation={res['rot_acc']:.1f}%  "
                    f"Neighbor={res['neigh_acc']:.1f}%"))
            except Exception as e:
                self.after(0, lambda: self._log_msg(f"✗ {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _plot_stage3(self):
        p   = self._puzzle
        sc  = self._scrambled
        res = self._stage3_result
        grid = res["best_grid"]
        rows, cols = p.rows, p.cols

        # Reconstruct solved tiles
        solved_tiles = []
        for r in range(rows):
            for c in range(cols):
                idx, rot = grid[r, c]
                solved_tiles.append(np.rot90(sc[idx], k=rot))

        self._fig.clear()
        self._fig.suptitle("Stage 3 — Αποτέλεσμα Επίλυσης",
                            color="#cdd6f4", fontsize=12, y=0.98)

        n = rows * cols
        # Layout: original | solved | metrics
        gs = gridspec.GridSpec(rows, cols * 2 + 1, figure=self._fig,
                               hspace=0.12, wspace=0.08,
                               left=0.02, right=0.97, top=0.90, bottom=0.12)

        # Original
        for idx in range(n):
            r, c = divmod(idx, cols)
            ax = self._fig.add_subplot(gs[r, c])
            ax.imshow(p.tiles[idx])
            ax.axis("off")
            if r == 0 and c == 0:
                ax.set_title("Original", fontsize=8, color="#89b4fa")

        # Solved
        for idx, tile in enumerate(solved_tiles):
            r, c = divmod(idx, cols)
            ax = self._fig.add_subplot(gs[r, cols + c])
            ax.imshow(tile)
            ax.axis("off")
            if r == 0 and c == 0:
                ax.set_title("Solved", fontsize=8, color="#a6e3a1")

        # Metrics bar chart (rightmost column, all rows merged)
        ax_m = self._fig.add_subplot(gs[:, cols * 2])
        metrics = {
            "Placement\nAccuracy": res["pos_acc"],
            "Rotation\nAccuracy":  res["rot_acc"],
            "Neighbor\nAccuracy":  res["neigh_acc"],
        }
        bar_colors = ["#89b4fa", "#a6e3a1", "#f9e2af"]
        bars = ax_m.barh(list(metrics.keys()), list(metrics.values()),
                         color=bar_colors, height=0.5)
        ax_m.set_xlim(0, 100)
        ax_m.set_xlabel("%", fontsize=8, color="#a6adc8")
        ax_m.set_facecolor("#181825")
        ax_m.tick_params(colors="#a6adc8", labelsize=8)
        for sp in ax_m.spines.values(): sp.set_color("#313244")
        for bar, val in zip(bars, metrics.values()):
            ax_m.text(val + 1, bar.get_y() + bar.get_height()/2,
                      f"{val:.1f}%", va="center", fontsize=9,
                      color="#cdd6f4", fontweight="bold")
        ax_m.set_title("Metrics", fontsize=9, color="#cdd6f4")

        # Time label
        self._fig.text(0.5, 0.04,
                       f"Χρόνος επίλυσης: {res['time']:.1f}s  |  "
                       f"Grid: {rows}×{cols}  |  Tiles: {n}",
                       ha="center", color="#6c7086", fontsize=9)
        self._draw()

    # ─────────────────────────────────────────────────────────────────
    #  ABLATION
    # ─────────────────────────────────────────────────────────────────
    def _run_ablation(self):
        if self._all_features is None:
            self._log_msg("✗ Τρέξε πρώτα Stage 2 & 3."); return

        self._log_msg("── Ablation Study ──")
        self._btns["ablation"].config(state="disabled")

        _, _, _, run_stage3_fn, ABLATION_CONFIGS, _, RESULTS_DIR, get_stage_paths = _import_project()
        from solver import JigsawSolver

        image_name = os.path.splitext(os.path.basename(self.image_path.get()))[0]
        ablation_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", image_name, "Stage_4_Ablation"
        )
        os.makedirs(ablation_dir, exist_ok=True)

        puzzle    = self._puzzle
        scrambled = self._scrambled
        feats     = self._all_features
        n_tiles   = len(scrambled)

        def _worker():
            results = {}
            for cfg_name, alpha_cfg in ABLATION_CONFIGS.items():
                self.after(0, lambda cn=cfg_name: self._log_msg(f"  → {cn}…"))
                solver = JigsawSolver(puzzle.rows, puzzle.cols, alpha_cfg)
                solver.all_tile_features = feats

                best_grid, best_score = None, -float("inf")
                for s_idx in range(n_tiles):
                    for s_rot in range(4):
                        g = solver._greedy_fill(s_idx, s_rot)
                        sc = solver.score_grid(g)
                        if sc > best_score:
                            best_score = sc; best_grid = g.copy()

                best_grid = solver._local_search(best_grid)
                pos, rot, neigh = solver.evaluate(
                    best_grid, puzzle.shuffled_indices, puzzle.applied_rotations)
                results[cfg_name] = {"pos": pos, "rot": rot, "neigh": neigh}

            self.after(0, lambda: self._plot_ablation(results))
            self.after(0, lambda: self._log_msg("✓ Ablation ολοκληρώθηκε."))
            self.after(0, lambda: self._enable_btn("ablation"))

        threading.Thread(target=_worker, daemon=True).start()

    def _plot_ablation(self, results):
        configs = list(results.keys())
        pos_v   = [results[c]["pos"]   for c in configs]
        rot_v   = [results[c]["rot"]   for c in configs]
        neigh_v = [results[c]["neigh"] for c in configs]

        self._fig.clear()
        self._fig.suptitle("Ablation Study — Σύγκριση Configurations (§6.1)",
                            color="#cdd6f4", fontsize=12, y=0.98)

        ax = self._fig.add_subplot(111)
        ax.set_facecolor("#181825")
        for sp in ax.spines.values(): sp.set_color("#313244")

        x = np.arange(len(configs))
        w = 0.25
        b1 = ax.bar(x - w, pos_v,   w, label="Placement %", color="#89b4fa", alpha=0.9)
        b2 = ax.bar(x,     rot_v,   w, label="Rotation %",  color="#a6e3a1", alpha=0.9)
        b3 = ax.bar(x + w, neigh_v, w, label="Neighbor %",  color="#f9e2af", alpha=0.9)

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                        f"{h:.1f}", ha="center", va="bottom",
                        fontsize=8, color="#cdd6f4")

        ax.set_xticks(x)
        ax.set_xticklabels(configs, color="#a6adc8", fontsize=10)
        ax.set_ylabel("Accuracy (%)", color="#a6adc8")
        ax.set_ylim(0, 115)
        ax.tick_params(colors="#a6adc8")
        ax.legend(facecolor="#313244", labelcolor="#cdd6f4", fontsize=9)
        ax.grid(axis="y", color="#313244", alpha=0.5)

        # Ανάδειξη νικητή
        scores = [pos_v[i] + rot_v[i] + neigh_v[i] for i in range(len(configs))]
        winner = configs[int(np.argmax(scores))]
        self._log_msg(f"  ★ Καλύτερο: '{winner}' (score={max(scores):.1f})")

        # Ερμηνεία κάτω
        interp = (
            "classical vs combined → αξία deep features  |  "
            "no_local vs combined → αξία Harris+SIFT  |  "
            "deep_only vs combined → αξία classical"
        )
        self._fig.text(0.5, 0.02, interp, ha="center",
                       color="#6c7086", fontsize=8)

        self._draw()


# ─────────────────────────────────────────────────────────────────────
#  ΒΟΗΘΗΤΙΚΗ: border γύρω από ax
# ─────────────────────────────────────────────────────────────────────
def _frame(ax, color):
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(1.5)


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = JigsawGUI()
    app.mainloop()
