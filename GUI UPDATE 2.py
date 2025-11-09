#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
ImageProcessingApp (v2, Top-5 Controls)
---------------------------------------
A compact PyQt5 GUI for batch BF/PL analysis with only the five most impactful controls:

1) Scale calibration (Known µm & Distance in pixels)  [treated as one control group]
2) Crystal scale range (σ_min–σ_max, in µm)
3) LoG threshold (sensitivity)
4) Minimum crystal area (µm²)
5) Watershed min_distance (px) for splitting cells

Processing modes (same ideas as original):
- Button 1: % Cells With Crystals (incidence)
- Button 2: % Area Crystals / Cells (coverage)
- Button 3: Number of Cells (cells only)

Author: (you)
"""

import os
import sys
import json
import zipfile
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless renders for saving plots
import matplotlib.pyplot as plt

# PyQt5 GUI
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QTextEdit, QFormLayout
)

# skimage / scipy
from skimage.color import rgb2gray
from skimage import exposure, morphology
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.feature import blob_log, peak_local_max
from skimage.segmentation import watershed
import scipy.ndimage as ndi

# Excel
from xlsxwriter import Workbook  # noqa: F401 (ensures engine availability)
import gc


def ensure_gray(img):
    """Convert BGR or RGB uint8 to float gray in [0,1]."""
    if img is None:
        return None
    if img.ndim == 3:
        g = rgb2gray(img)
    else:
        # assume already grayscale-like
        g = img.astype(np.float32)
        if g.max() > 1.0:
            g = g / 255.0
    return g


def bilateral_uint8(gray01, d=9, sC=75, sS=75):
    """Apply OpenCV bilateral to a [0,1] float image, return uint8 [0..255]."""
    g8 = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)
    out = cv2.bilateralFilter(g8, d, sC, sS)
    return out


def area_um2(area_px, um_per_px):
    return float(area_px) * (um_per_px ** 2)


def sigma_um_to_px(sigma_um, um_per_px):
    """Convert LoG scale in µm to sigma in pixels."""
    # σ [px] = σ [µm] / (µm/px)
    if um_per_px <= 0:
        return sigma_um  # fallback
    return float(sigma_um) / float(um_per_px)


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.bf_folder = ""
        self.pl_folder = ""
        self.output_folder = ""

        # ---- Top-5 controls (with sensible defaults) ----
        self.known_um = 40.0           # µm (from scale bar text)
        self.distance_px = 400.0       # px (pixels measured across that bar)
        self.sigma_min_um = 0.6        # µm
        self.sigma_max_um = 3.0        # µm
        self.log_threshold = 0.02      # LoG response
        self.min_crystal_area_um2 = 1.0
        self.ws_min_distance_px = 5    # watershed seed spacing (BF split)

        self.initUI()

    # ---------------- GUI ----------------
    def initUI(self):
        self.setWindowTitle("Batch Image Processing – v2 (Top-5 Controls)")
        self.resize(720, 560)

        root = QVBoxLayout()

        # Folder selectors
        folders_box = QHBoxLayout()
        self.bf_label = QLabel("BF Folder: Not selected")
        self.pl_label = QLabel("PL Folder: Not selected")
        self.out_label = QLabel("Output Folder: Not selected")
        btn_bf = QPushButton("Select BF Folder")
        btn_pl = QPushButton("Select PL Folder")
        btn_out = QPushButton("Select Output Folder")
        btn_bf.clicked.connect(self.select_bf_folder)
        btn_pl.clicked.connect(self.select_pl_folder)
        btn_out.clicked.connect(self.select_output_folder)
        fcol = QVBoxLayout()
        fcol.addWidget(self.bf_label)
        fcol.addWidget(self.pl_label)
        fcol.addWidget(self.out_label)
        fbtns = QVBoxLayout()
        fbtns.addWidget(btn_bf)
        fbtns.addWidget(btn_pl)
        fbtns.addWidget(btn_out)
        folders_box.addLayout(fcol, 2)
        folders_box.addLayout(fbtns, 1)

        # Controls (Top-5)
        form = QFormLayout()
        self.e_known_um = QLineEdit(str(self.known_um))
        self.e_distance_px = QLineEdit(str(self.distance_px))
        self.e_sigma_min_um = QLineEdit(str(self.sigma_min_um))
        self.e_sigma_max_um = QLineEdit(str(self.sigma_max_um))
        self.e_log_threshold = QLineEdit(str(self.log_threshold))
        self.e_min_cr_area = QLineEdit(str(self.min_crystal_area_um2))
        self.e_ws_min_dist = QLineEdit(str(self.ws_min_distance_px))

        form.addRow(QLabel("<b>Calibration</b> (Known µm & Distance in px)"))
        form.addRow("Known distance (µm):", self.e_known_um)
        form.addRow("Distance in pixels:", self.e_distance_px)

        form.addRow(QLabel("<b>Crystals (PL)</b>"))
        form.addRow("σ_min (µm):", self.e_sigma_min_um)
        form.addRow("σ_max (µm):", self.e_sigma_max_um)
        form.addRow("LoG threshold:", self.e_log_threshold)
        form.addRow("Min crystal area (µm²):", self.e_min_cr_area)

        form.addRow(QLabel("<b>Cells (BF)</b>"))
        form.addRow("Watershed min_distance (px):", self.e_ws_min_dist)

        # Action buttons
        btns = QHBoxLayout()
        self.btn_run1 = QPushButton("Run – % Cells With Crystals")
        self.btn_run2 = QPushButton("Run – % Area Crystals / Cells")
        self.btn_run3 = QPushButton("Run – Number of Cells")
        self.btn_run1.clicked.connect(lambda: self.run(mode=1))
        self.btn_run2.clicked.connect(lambda: self.run(mode=2))
        self.btn_run3.clicked.connect(lambda: self.run(mode=3))
        btns.addWidget(self.btn_run1)
        btns.addWidget(self.btn_run2)
        btns.addWidget(self.btn_run3)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # Assemble
        root.addLayout(folders_box)
        root.addSpacing(8)
        root.addLayout(form)
        root.addSpacing(8)
        root.addLayout(btns)
        root.addWidget(self.log)
        self.setLayout(root)

    # -------------- Folders --------------
    def select_bf_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select BF Folder")
        if folder:
            self.bf_folder = folder
            self.bf_label.setText(f"BF Folder: {folder}")

    def select_pl_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PL Folder")
        if folder:
            self.pl_folder = folder
            self.pl_label.setText(f"PL Folder: {folder}")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.out_label.setText(f"Output Folder: {folder}")

    # -------------- Util --------------
    def get_params(self):
        try:
            known_um = float(self.e_known_um.text())
            distance_px = float(self.e_distance_px.text())
            sigma_min_um = float(self.e_sigma_min_um.text())
            sigma_max_um = float(self.e_sigma_max_um.text())
            log_thr = float(self.e_log_threshold.text())
            min_cr_um2 = float(self.e_min_cr_area.text())
            ws_min = int(float(self.e_ws_min_dist.text()))
        except Exception:
            QMessageBox.warning(self, "Input error", "Please enter valid numeric values.")
            return None

        if distance_px <= 0 or known_um <= 0:
            QMessageBox.warning(self, "Calibration error", "Known µm and Distance px must be positive.")
            return None

        if sigma_max_um < sigma_min_um:
            QMessageBox.warning(self, "Scale range", "σ_max must be ≥ σ_min.")
            return None

        return dict(
            known_um=known_um, distance_px=distance_px,
            sigma_min_um=sigma_min_um, sigma_max_um=sigma_max_um,
            log_threshold=log_thr, min_crystal_area_um2=min_cr_um2,
            ws_min_distance_px=ws_min
        )

    # -------------- Main run --------------
    def run(self, mode=1):
        p = self.get_params()
        if p is None:
            return
        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            QMessageBox.information(self, "Select folders", "Please select BF, PL and Output folders first.")
            return

        os.makedirs(self.output_folder, exist_ok=True)

        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.lower().endswith(".tif")])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.lower().endswith(".tif")])
        if len(bf_files) != len(pl_files) or len(bf_files) == 0:
            QMessageBox.warning(self, "Pairing", "Mismatch or no files in BF/PL.")
            return

        # Calibration
        um_per_px = p["known_um"] / p["distance_px"]  # correct physics: µm/px

        sigma_min_px = sigma_um_to_px(p["sigma_min_um"], um_per_px)
        sigma_max_px = sigma_um_to_px(p["sigma_max_um"], um_per_px)
        if sigma_min_px <= 0 or sigma_max_px <= 0:
            QMessageBox.warning(self, "Scale", "Converted sigma (px) must be positive.")
            return

        # Summary collectors
        summary_rows = []
        all_output_files = []

        for bf_file, pl_file in zip(bf_files, pl_files):
            bf_path = os.path.join(self.bf_folder, bf_file)
            pl_path = os.path.join(self.pl_folder, pl_file)

            imageA = cv2.imread(bf_path, cv2.IMREAD_COLOR)
            imageB = cv2.imread(pl_path, cv2.IMREAD_COLOR)
            if imageA is None or imageB is None:
                self.log.append(f"Skipping {bf_file}/{pl_file}: could not load.")
                continue

            if imageA.shape[:2] != imageB.shape[:2]:
                self.log.append(f"Warning: {bf_file}/{pl_file} have different sizes; results may be misaligned.")

            # ------------ BF: cell mask ------------
            grayA = ensure_gray(imageA)
            # Mask out a small bottom-right region to remove scale bar (same % as original)
            h, w = grayA.shape
            mask = np.ones_like(grayA, dtype=bool)
            mask[h - int(0.015*h):, w - int(0.025*w):] = False
            grayA = np.where(mask, grayA, 0.0)

            # CLAHE + bilateral (fixed – not exposed)
            grayA = exposure.equalize_adapthist(grayA, clip_limit=0.01)
            denA = bilateral_uint8(grayA, d=9, sC=75, sS=75)

            thr = threshold_otsu(denA)
            binA = (denA < thr).astype(np.uint8)  # cells are darker than background
            # Morphology (fixed – not exposed)
            binA = morphology.remove_small_objects(binA.astype(bool), min_size=1600)
            binA = morphology.remove_small_holes(binA, area_threshold=10000)
            binA = morphology.opening(binA)
            binA = (binA > 0).astype(np.uint8)

            # Label + optional watershed split (exposed min_distance)
            lblA = label(binA)
            props = regionprops(lblA)
            # build geo seeds using distance transform
            dist = ndi.distance_transform_edt(binA)
            coords = peak_local_max(dist, min_distance=p["ws_min_distance_px"], labels=binA, exclude_border=False)
            seed = np.zeros_like(binA, dtype=np.int32)
            for i, (yy, xx) in enumerate(coords, start=1):
                seed[yy, xx] = i
            lblA = watershed(-dist, seed, mask=binA)
            props = regionprops(lblA)

            # Export cells (areas)
            cell_rows = []
            for r in props:
                a_px = r.area
                if a_px <= 800:  # keep same export filter as original
                    continue
                cell_rows.append(dict(
                    Region_Label=r.label,
                    Region_Area_pixels=int(a_px),
                    Region_Area_um2=area_um2(a_px, um_per_px)
                ))
            df_cells = pd.DataFrame(cell_rows)
            total_cells = len(df_cells)
            total_cell_area_um2 = df_cells["Region_Area_um2"].sum() if not df_cells.empty else 0.0

            # ------------ PL: crystals via LoG ------------
            grayB = ensure_gray(imageB)
            grayB = exposure.equalize_adapthist(grayB, clip_limit=0.01)
            denB = bilateral_uint8(grayB, d=9, sC=75, sS=75)  # uint8

            # blob_log expects float [0..1]
            fB = denB.astype(np.float32) / 255.0
            blobs = blob_log(fB,
                             min_sigma=max(0.5, sigma_min_px),
                             max_sigma=max(sigma_min_px + 0.5, sigma_max_px),
                             num_sigma=6,
                             threshold=p["log_threshold"])
            # Each blob: (y, x, sigma)
            mask_cr = np.zeros_like(denB, dtype=np.uint8)
            for (yy, xx, ss) in blobs:
                r = int(max(1, ss))  # simple radius from sigma (kept like original spirit)
                cv2.circle(mask_cr, (int(xx), int(yy)), r, 255, -1)

            # Keep only crystals overlapping cells
            overlap = ((mask_cr > 0) & (lblA > 0)).astype(np.uint8)

            # Label crystals and filter by min area (µm²)
            lblC = label(overlap)
            propsC = regionprops(lblC)
            min_area_px = (p["min_crystal_area_um2"] / (um_per_px ** 2)) if um_per_px > 0 else 0.0

            cry_rows = []
            for r in propsC:
                if r.area < min_area_px:
                    continue
                # map to cell by maximal overlap
                valA = lblA[r.coords[:, 0], r.coords[:, 1]]
                valA = valA[valA > 0]
                if valA.size == 0:
                    best_cell = None
                else:
                    labels, counts = np.unique(valA, return_counts=True)
                    best_cell = int(labels[np.argmax(counts)])
                cry_rows.append(dict(
                    Region_Label=int(r.label),
                    Associated_Cell=best_cell if best_cell is not None else "",
                    Overlap_pixels=int(r.area),
                    Region_Area_pixels=int(r.area),
                    Region_Area_um2=area_um2(r.area, um_per_px)
                ))
            df_cr = pd.DataFrame(cry_rows)

            # ------------- Summaries by mode -------------
            base = os.path.splitext(bf_file)[0]
            pair_tag = base

            if mode == 1:
                # % cells with ≥1 crystal
                cells_with_crystals = df_cr["Associated_Cell"].replace("", np.nan).dropna().astype(int).nunique() if not df_cr.empty else 0
                pct = (cells_with_crystals / total_cells * 100.0) if total_cells > 0 else 0.0
                summary_rows.append(dict(
                    Image=pair_tag,
                    total_cells=total_cells,
                    cells_with_crystals=cells_with_crystals,
                    pct_cells_with_crystals=pct
                ))
            elif mode == 2:
                # % area crystals / cells
                total_cr_area_um2 = df_cr["Region_Area_um2"].sum() if not df_cr.empty else 0.0
                pct = (total_cr_area_um2 / total_cell_area_um2 * 100.0) if total_cell_area_um2 > 0 else 0.0
                summary_rows.append(dict(
                    Image=pair_tag,
                    total_cell_area_um2=total_cell_area_um2,
                    total_crystal_area_um2=total_cr_area_um2,
                    pct_area_crystals_cells=pct
                ))
            elif mode == 3:
                # Number of cells only
                summary_rows.append(dict(
                    Image=pair_tag,
                    total_cells=total_cells
                ))

            # ------------- Save per-pair Excel -------------
            xlsx_path = os.path.join(self.output_folder, f"{pair_tag}_Results.xlsx")
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                df_cells.to_excel(writer, sheet_name="Cells", index=False)
                df_cr.to_excel(writer, sheet_name="Crystals", index=False)
            all_output_files.append(xlsx_path)

            # ------------- Quick annotated overlay -------------
            overlay = imageA.copy()
            # show cell labels (centroids)
            for r in regionprops(lblA):
                (yy, xx) = r.centroid
                cv2.putText(overlay, str(r.label), (int(xx), int(yy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            # show crystal mask in yellow
            ymask = np.dstack([np.zeros_like(mask_cr), mask_cr, mask_cr])
            overlay = cv2.addWeighted(overlay, 1.0, ymask, 0.5, 0.0)
            ann_path = os.path.join(self.output_folder, f"{pair_tag}_Annotated.png")
            cv2.imwrite(ann_path, overlay)
            all_output_files.append(ann_path)

            # ------------- Histograms (diagnostic) -------------
            # BF intensity histogram
            fig = plt.figure(figsize=(6, 4))
            plt.hist(denA.ravel(), bins=256, range=[0, 255])
            plt.title("BF Intensity Histogram")
            plt.tight_layout()
            histA = os.path.join(self.output_folder, f"{pair_tag}_Histogram_BF.png")
            plt.savefig(histA, dpi=160)
            plt.close(fig)
            all_output_files.append(histA)

            # PL intensity histogram
            fig = plt.figure(figsize=(6, 4))
            plt.hist(denB.ravel(), bins=256, range=[0, 255])
            plt.title("PL Intensity Histogram")
            plt.tight_layout()
            histB = os.path.join(self.output_folder, f"{pair_tag}_Histogram_PL.png")
            plt.savefig(histB, dpi=160)
            plt.close(fig)
            all_output_files.append(histB)

            self.log.append(f"Processed: {bf_file} / {pl_file}")
            QApplication.processEvents()
            del grayA, denA, grayB, denB, lblA, lblC
            gc.collect()

        # Save summary table and simple plot per mode
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            sum_xlsx = os.path.join(self.output_folder, "Summary.xlsx")
            df_sum.to_excel(sum_xlsx, index=False)

            # Simple plot (if percentage column exists)
            ycol = None
            if mode == 1 and "pct_cells_with_crystals" in df_sum.columns:
                ycol = "pct_cells_with_crystals"
                title = "% Cells With Crystals"
                ylabel = "%"
            elif mode == 2 and "pct_area_crystals_cells" in df_sum.columns:
                ycol = "pct_area_crystals_cells"
                title = "% Area Crystals / Cells"
                ylabel = "%"

            if ycol is not None:
                fig = plt.figure(figsize=(7, 4))
                xs = np.arange(len(df_sum))
                plt.plot(xs, df_sum[ycol].values, marker="o")
                plt.xticks(xs, df_sum["Image"].astype(str), rotation=45, ha="right")
                plt.title(title)
                plt.ylabel(ylabel)
                plt.tight_layout()
                plot_path = os.path.join(self.output_folder, "Summary.png")
                plt.savefig(plot_path, dpi=180)
                plt.close(fig)
                all_output_files.append(plot_path)

        # Zip images/plots (optional)
        if all_output_files:
            zip_path = os.path.join(self.output_folder, "All_Exports.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in all_output_files:
                    if os.path.exists(f):
                        zf.write(f, arcname=os.path.basename(f))

        self.log.append("Done.")

# ---------------- Entry ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ImageProcessingApp()
    w.show()
    sys.exit(app.exec_())


# In[ ]:




