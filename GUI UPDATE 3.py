#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ─── Standard Library Imports ─────────────────────────────
import os  # Operating system interaction
import sys  # Access to system-specific parameters and functions
import json  # Reading and writing JSON configuration
import zipfile  # Handling ZIP file creation

# ─── Third-party Imports ──────────────────────────────────
import cv2  # OpenCV for image processing
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and tables
import openpyxl  # Excel file I/O
import matplotlib.pyplot as plt  # Plotting
import matplotlib  # Matplotlib config
matplotlib.use("Qt5Agg")  # Use Qt5Agg backend for GUI support

# ─── PyQt5 GUI Components ────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QTextEdit, QInputDialog, QComboBox, QFormLayout,
    QDoubleSpinBox, QGroupBox
)

# ─── Image I/O ───────────────────────────────────────────
import imageio.v2 as imageio  # Image reading/writing (legacy v2 API)

# ─── Skimage Modules for Image Processing ────────────────
from skimage.measure import label, regionprops  # Region labeling
from skimage.filters import threshold_li, threshold_otsu, threshold_isodata  # Threshold methods
from skimage.feature import blob_log
from skimage import data, filters, measure, morphology  # Generic image ops
from skimage.color import rgb2gray  # Convert RGB to grayscale
from skimage.morphology import (
    opening, remove_small_objects, remove_small_holes, disk
)  # Morphological ops
from skimage import util, exposure, color  # Image enhancement and color ops
from skimage.feature import peak_local_max  # Peak detection
from skimage.segmentation import (
    morphological_chan_vese, slic, active_contour,
    watershed, random_walker
)  # Various segmentation algorithms
from skimage.io import imread  # Image reading
from skimage.transform import resize  # Image resizing
from skimage import draw  # Drawing shapes

# ─── SciPy for Advanced Processing ───────────────────────
import scipy.ndimage as ndi  # Multidimensional processing
from scipy.ndimage import distance_transform_edt, label as ndi_label  # Distance transforms and labeling
from scipy import ndimage  # General ndimage support
from scipy.signal import find_peaks  # Signal peak detection

# ─── Machine Learning ─────────────────────────────────────
from sklearn.cluster import KMeans  # Clustering (e.g., for region grouping)

# ─── Excel Writing ───────────────────────────────────────
from xlsxwriter import Workbook  # Advanced Excel writing

# ─── Qt Event Processing ─────────────────────────────────
QApplication.processEvents()  # Process any pending GUI events

# ─── Threading & Event Control ───────────────────────────
from threading import Event  # Used to signal stopping of processing

# ─── Utilities ────────────────────────────────────────────
from collections import defaultdict  # Dictionary that creates default values automatically

import gc

# ─────────────────────────────────────────────────────────
# GUI Application Class for Image Processing
class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  # Set up GUI layout

        # Default scale mapping (µm to pixels)
        self.um_to_px_map = {
            "10": 2.5,
            "20": 1.25,
            "40": 10,
            "50": 25,
            "100": 10,
            "200": 1.7
        }

        # Initialize folder paths and control flags
        self.bf_folder = ""
        self.pl_folder = ""
        self.output_folder = ""
        self.processing_active = False  # Track if a process is currently running
        self.stop_event = Event()  # Event to handle stop signal

        self.load_scale_settings()  # Load saved scale mappings

    def initUI(self):
        # Create the GUI layout
        layout = QVBoxLayout()

        # ---- Parameter Panel (4 knobs) ----
        params_group = QGroupBox("Detection parameters")
        form = QFormLayout(params_group)

        # 1) Expected crystal diameter (µm)
        self.spin_crystal_d_um = QDoubleSpinBox()
        self.spin_crystal_d_um.setRange(0.5, 50.0)
        self.spin_crystal_d_um.setSingleStep(0.5)
        self.spin_crystal_d_um.setValue(4.0)
        form.addRow("Expected crystal diameter (µm)", self.spin_crystal_d_um)

        # 2) Crystal detection sensitivity (blob threshold)
        self.spin_crystal_thresh = QDoubleSpinBox()
        self.spin_crystal_thresh.setRange(0.005, 0.08)
        self.spin_crystal_thresh.setSingleStep(0.005)
        self.spin_crystal_thresh.setDecimals(3)
        self.spin_crystal_thresh.setValue(0.02)
        form.addRow("Crystal detection sensitivity", self.spin_crystal_thresh)

        # 3) Cell split aggressiveness (µm)
        self.spin_split_um = QDoubleSpinBox()
        self.spin_split_um.setRange(0.5, 20.0)
        self.spin_split_um.setSingleStep(0.5)
        self.spin_split_um.setValue(3.0)
        form.addRow("Cell split aggressiveness (µm)", self.spin_split_um)

        # 4) Cell mask threshold bias
        self.spin_thresh_bias = QDoubleSpinBox()
        self.spin_thresh_bias.setRange(0.80, 1.20)
        self.spin_thresh_bias.setSingleStep(0.01)
        self.spin_thresh_bias.setDecimals(2)
        self.spin_thresh_bias.setValue(1.00)
        form.addRow("Cell mask threshold bias", self.spin_thresh_bias)

        layout.addWidget(params_group)

        # Label and input for pixel distance
        self.pixel_distance_label = QLabel("Distance in pixels:")
        self.pixel_distance_input = QLineEdit()
        self.pixel_distance_input.setText("NOT VALUE")

        # Label and combo box for known µm distances
        self.known_um_label = QLabel("Known distance (µm):")
        self.known_um_combo = QComboBox()
        self.known_um_combo.setEditable(True
        )
        self.known_um_combo.addItems(["40", "100"])
        self.known_um_combo.setCurrentText("NOT VALUE")
        self.known_um_combo.setInsertPolicy(QComboBox.InsertAtBottom)
        self.known_um_combo.lineEdit().editingFinished.connect(self.on_custom_um_entered)
        self.known_um_combo.currentIndexChanged.connect(self.update_pixel_distance)

        # Labels for folder selection display
        self.bf_label = QLabel("BF Folder: Not selected")
        self.pl_label = QLabel("PL Folder: Not selected")
        self.output_label = QLabel("Output Folder: Not selected")

        # Buttons for actions and controls
        self.set_scale_button = QPushButton("Set µm to px Scale")
        self.delete_scale_button = QPushButton("Delete Selected Scale")
        self.bf_button = QPushButton("Select BF Folder")
        self.pl_button = QPushButton("Select PL Folder")
        self.output_button = QPushButton("Select Output Folder")
        self.process_button = QPushButton("Number of crystals")
        self.process_button_2 = QPushButton("Areas")
        self.process_button_3 = QPushButton("Number of cells")
        self.stop_button = QPushButton("Stop Processing")
        self.restart_button = QPushButton("Restart Processing")

        # Log output window
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # Connect button actions to their corresponding methods
        self.set_scale_button.clicked.connect(self.set_known_um_and_px)
        self.delete_scale_button.clicked.connect(self.delete_selected_scale)
        self.bf_button.clicked.connect(self.select_bf_folder)
        self.pl_button.clicked.connect(self.select_pl_folder)
        self.output_button.clicked.connect(self.select_output_folder)
        self.process_button.clicked.connect(self.start_processing)
        self.process_button_2.clicked.connect(self.start_processing_2)
        self.process_button_3.clicked.connect(self.start_processing_3)
        self.stop_button.clicked.connect(self.stop_processing)
        self.restart_button.clicked.connect(self.restart_processing)

        # Add widgets to the GUI layout
        layout.addWidget(self.set_scale_button)
        layout.addWidget(self.delete_scale_button)
        layout.addWidget(self.pixel_distance_label)
        layout.addWidget(self.pixel_distance_input)
        layout.addWidget(self.known_um_label)
        layout.addWidget(self.known_um_combo)
        layout.addWidget(self.bf_label)
        layout.addWidget(self.bf_button)
        layout.addWidget(self.pl_label)
        layout.addWidget(self.pl_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.process_button_2)
        layout.addWidget(self.process_button_3)
        layout.addWidget(self.log_output)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.restart_button)

        # Finalize window settings
        self.setLayout(layout)
        self.setWindowTitle("Batch Image Processing")
        self.resize(500, 400)

    def log(self, message):
        # Append a log message to the log output display (likely a QTextEdit or QListWidget)
        self.log_output.append(message)

    # ---- Helpers for parameterized detection ----
    def _sigma_range_from_expected_d_um(self, expected_d_um, pixel_to_um):
        """
        Map expected diameter (µm) -> (min_sigma_px, max_sigma_px, num_sigma)
        For LoG: diameter d ≈ 2√2·σ  => σ ≈ d / (2√2).
        Search around expected size with a ±60% band.
        """
        if pixel_to_um <= 0:
            return (1, 8, 5)  # fallback in pixels
        sigma_expected_px = (expected_d_um / (2.0 * np.sqrt(2.0))) / pixel_to_um
        min_sigma = max(0.5, 0.4 * sigma_expected_px)
        max_sigma = max(min_sigma + 0.5, 1.6 * sigma_expected_px)
        return (min_sigma, max_sigma, 5)

    def _auto_threshold_with_bias(self, gray_uint8, bias=1.0):
        """
        Try Otsu with bias; if the mask is implausibly small/large,
        try Li and IsoData and pick the one whose foreground ratio is near 0.25.
        """
        import skimage.filters as _f

        def make_mask(t):
            t_adj = int(np.clip(t * bias, 0, 255))
            return (gray_uint8 < t_adj).astype(np.uint8)

        methods = [
            ("otsu", _f.threshold_otsu),
            ("li", _f.threshold_li),
            ("isodata", _f.threshold_isodata),
        ]

        # First pass with Otsu
        t_otsu = methods[0][1](gray_uint8)
        mask = make_mask(t_otsu)
        frac = mask.mean()

        # Keep if plausible
        if 0.02 <= frac <= 0.60:
            return mask * 255

        # Otherwise test Li and IsoData, pick closest to 0.25
        candidates = []
        for name, func in methods:
            t = func(gray_uint8)
            m = make_mask(t)
            candidates.append((name, m, abs(m.mean() - 0.25)))
        best = min(candidates, key=lambda x: x[2])[1]
        return best * 255

    def on_custom_um_entered(self):
        # Handle user entering a custom µm value in the combo box
        text = self.known_um_combo.currentText().strip()
        if text not in [self.known_um_combo.itemText(i) for i in range(self.known_um_combo.count())]:
            self.known_um_combo.addItem(text)

    def update_pixel_distance(self):
        # Update the pixel distance input field based on the selected scale
        text = self.known_um_combo.currentText()
        if text in self.um_to_px_map:
            self.pixel_distance_input.setText(str(self.um_to_px_map[text]))
        else:
            self.pixel_distance_input.clear()

    def select_bf_folder(self):
        # Prompt user to select a folder for BF (Brightfield) images
        self.bf_folder = QFileDialog.getExistingDirectory(self, "Select BF Folder")
        self.bf_label.setText(f"BF Folder: {self.bf_folder}")

    def select_pl_folder(self):
        # Prompt user to select a folder for PL (Polarized Light) images
        self.pl_folder = QFileDialog.getExistingDirectory(self, "Select PL Folder")
        self.pl_label.setText(f"PL Folder: {self.pl_folder}")

    def select_output_folder(self):
        # Prompt user to select a folder to save outputs
        self.output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        self.output_label.setText(f"Output Folder: {self.output_folder}")

    def stop_processing(self):
        # Set the stop event flag to signal that processing should stop
        self.stop_event.set()
        self.log("Stopping process...")

    def restart_processing(self):
        # Stop current process and then start Script 3 again
        self.stop_processing()
        self.log("Restarting processing...")
        self.start_processing_3()

    def save_scale_settings(self):
        # Save the scale mapping dictionary to a JSON file
        with open('scale_map.json', 'w') as f:
            json.dump(self.um_to_px_map, f)

    def load_scale_settings(self):
        # Load scale mapping from a JSON file; use defaults if not found
        try:
            with open('scale_map.json', 'r') as f:
                self.um_to_px_map = json.load(f)
        except FileNotFoundError:
            self.um_to_px_map = {
                "10": 2.5,
                "20": 1.25,
                "40": 10,
                "50": 25,
                "100": 10,
                "200": 1.7
            }

        # Clear and update the known µm combo box with loaded values
        self.known_um_combo.clear()
        self.known_um_combo.addItems(self.um_to_px_map.keys())

    def set_known_um_and_px(self):
        # Prompt user to input a known real-world micrometer value
        known_um, ok1 = QInputDialog.getDouble(self, "Known µm", "Enter known micrometer value:", decimals=6)
        if not ok1:
            return

        # Prompt user to input the corresponding pixel distance
        distance_px, ok2 = QInputDialog.getDouble(self, "Distance in Pixels", "Enter distance in pixels:", decimals=6)
        if not ok2 or distance_px == 0:
            return

        # Calculate µm per pixel ratio
        um_per_px = known_um / distance_px
        name = f"{known_um}"

        # Save this new scale in the map and refresh the combo box
        self.um_to_px_map[name] = um_per_px
        self.save_scale_settings()
        self.load_scale_settings()
        self.known_um_combo.setCurrentText(name)

        # Notify user that scale was saved
        QMessageBox.information(self, "Saved", f"Added mapping '{name}' = {um_per_px:.6f} µm/px")

    def load_scales_from_json(self):
        # Load scales from a predefined JSON file, fallback to default if failed
        try:
            with open("scales.json", "r") as f:
                scales = json.load(f)
            return scales
        except Exception:
            return {"10": 2.5, "20": 1.25, "40": 10, "50": 25, "100": 10, "200": 1.7}

    def add_new_scale(self, scale_name, scale_value):
        # Add new scale mapping and save it
        self.um_to_px_map[scale_name] = scale_value
        self.save_scale_settings()

    def delete_selected_scale(self):
        # Delete selected scale from the combo box and mapping
        selected_scale = self.known_um_combo.currentText()
        if selected_scale in self.um_to_px_map and selected_scale not in ["10", "20", "40", "50", "100", "200"]:
            confirm = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete the scale '{selected_scale}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if confirm == QMessageBox.Yes:
                del self.um_to_px_map[selected_scale]
                self.save_scale_settings()
                self.load_scale_settings()
                self.pixel_distance_input.clear()
                self.known_um_combo.setCurrentText("NOT VALUE")
                self.log(f"Deleted scale '{selected_scale}'")
        else:
            QMessageBox.warning(self, "Not Found", f"The scale '{selected_scale}' can not be delete.")

    def _compute_pixel_to_um(self):
        distance_in_px = float(self.pixel_distance_input.text())
        known_um = float(self.known_um_combo.currentText())
        if distance_in_px == 0:
            raise ValueError("Distance in pixels cannot be zero.")
        # pixel_to_um: micrometers per pixel
        return known_um / distance_in_px

    # --------------- Pipeline 1: Number of crystals -----------------
    def start_processing(self):
        self.processing_active = True
        self.stop_event.clear()

        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        try:
            pixel_to_um = self._compute_pixel_to_um()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for distance in pixels and known µm.")
            return None

        os.makedirs(self.output_folder, exist_ok=True)

        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.endswith('.tif')])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.endswith('.tif')])

        if len(bf_files) != len(pl_files):
            raise ValueError("Mismatch in the number of BF and PL .tif files.")

        all_output_files = []
        summary_rows = []
        min_cell_area_um2 = 20.0  # filter very small cell fragments

        for bf_file, pl_file in zip(bf_files, pl_files):
            print(f"Processing: {bf_file} and {pl_file}")
            if self.stop_event.is_set():
                self.log("Processing stopped.")
                return

            self.log(f"Processing {bf_file} and {pl_file}...")

            bf_image_path = os.path.join(self.bf_folder, bf_file)
            pl_image_path = os.path.join(self.pl_folder, pl_file)
            imageA = cv2.imread(bf_image_path)
            imageB = cv2.imread(pl_image_path)

            if imageA is None or imageB is None:
                print(f"Skipping {bf_file} or {pl_file}: Unable to load image.")
                continue

            # ---- BF preprocessing & threshold (auto+ bias) ----
            grayA = rgb2gray(imageA)
            h, w = grayA.shape
            crop_margin_h = int(0.015 * h)
            crop_margin_w = int(0.025 * w)
            scale_mask = np.ones_like(grayA, dtype=bool)
            scale_mask[h - crop_margin_h:, w - crop_margin_w:] = False
            grayA = grayA * scale_mask
            grayA = exposure.equalize_adapthist(grayA)
            grayA_u8 = cv2.bilateralFilter((np.clip(grayA, 0, 1) * 255).astype(np.uint8), 9, 75, 75)

            bias = float(self.spin_thresh_bias.value())
            binary_A = self._auto_threshold_with_bias(grayA_u8, bias=bias)

            binary_A = morphology.remove_small_objects(binary_A.astype(bool), min_size=1600)
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=10000)
            binary_A = morphology.opening(binary_A)
            binary_A = (binary_A > 0).astype(np.uint8) * 255

            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            # Exclude cropped area regions
            crop_start_row = h - crop_margin_h
            crop_start_col = w - crop_margin_w
            crop_mask = np.zeros_like(region_labels_A, dtype=bool)
            crop_mask[crop_start_row:, crop_start_col:] = True

            filtered_labels = []
            for region in region_props_A:
                region_mask = (region_labels_A == region.label)
                if not np.any(region_mask & crop_mask):
                    filtered_labels.append(region.label)

            new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
            label_counter = 1
            for lbl in filtered_labels:
                new_label_img[region_labels_A == lbl] = label_counter
                label_counter += 1

            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            areas = [region.area for region in region_props_A] or [1]
            media_area = np.median(areas)
            min_area = np.min(areas)
            std_area = np.std(areas)
            average = min_area + std_area

            # Save histogram of areas
            plt.figure(figsize=(8, 5))
            plt.hist(areas, bins=20, edgecolor='black')
            plt.title("Histogram of Region Areas")
            plt.xlabel("Area (pixels)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            hist_areas_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_Areas.png")
            plt.savefig(hist_areas_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            all_output_files.append(hist_areas_image_path)

            # Split big regions with watershed, min_distance driven by µm control
            for region in region_props_A:
                if region.area < average:
                    new_label_img[region.slice][region.image] = label_counter
                    label_counter += 1
                else:
                    region_mask = np.zeros_like(region_labels_A, dtype=np.uint8)
                    region_mask[region.slice][region.image] = 1
                    distance = ndi.distance_transform_edt(region_mask)
                    split_um = float(self.spin_split_um.value())
                    min_dist_px = max(1, int(round(split_um / pixel_to_um)))
                    coordinates = peak_local_max(distance, labels=region_mask, min_distance=min_dist_px)
                    local_maxi = np.zeros_like(distance, dtype=bool)
                    if coordinates.size > 0:
                        local_maxi[tuple(coordinates.T)] = True
                    markers = label(local_maxi)
                    labels_ws = watershed(-distance, markers, mask=region_mask)
                    for ws_label in np.unique(labels_ws):
                        if ws_label == 0:
                            continue
                        mask_ws = labels_ws == ws_label
                        new_label_img[mask_ws] = label_counter
                        label_counter += 1

            region_labels_A = new_label_img
            region_labels_A = label(region_labels_A > 0)
            region_props_A = regionprops(region_labels_A)

            if binary_A.shape != grayA_u8.shape:
                binary_A = resize(binary_A, grayA_u8.shape, order=0, preserve_range=True, anti_aliasing=False)

            # Save segmentation label image (cast to uint16 for safety)
            seg_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented.png")
            cv2.imwrite(seg_path, region_labels_A.astype(np.uint16))
            all_output_files.append(seg_path)

            # Annotate labels
            overlay_image = cv2.cvtColor((binary_A > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for region in regionprops(region_labels_A):
                y, x = region.centroid
                cv2.putText(overlay_image, str(region.label), (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            annotated_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented_Annotated.png")
            cv2.imwrite(annotated_path, overlay_image)
            all_output_files.append(annotated_path)

            # Valid detected regions mask
            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (
                        region_labels_A[min_row:max_row, min_col:max_col] == prop.label
                    )
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255

            # Region statistics
            region_area = pd.DataFrame({
                "Region_Label": [r.label for r in region_props_A],
                "Region_Area (pixels)": [r.area for r in region_props_A],
                "Region_Area (µm²)": [r.area * (pixel_to_um ** 2) for r in region_props_A]
            })

            region_area_df = region_area[region_area["Region_Area (µm²)"] > min_cell_area_um2]
            total_area = region_area_df["Region_Area (µm²)"].sum()
            total_cells = region_area_df["Region_Label"].count()

            # Append summary rows
            region_area_df.loc["Total Area"] = ["", "Total Area", total_area]
            region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

            # BF intensity histogram
            plt.figure(figsize=(8, 6))
            plt.hist(grayA_u8.ravel(), bins=256, range=[0, 255], alpha=0.7)
            plt.title('Histogram of Pixel Intensities (BF)')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            all_output_files.append(hist_cells_image_path)

            # ---- PL image for crystals (blob_log with parameterized size/sensitivity) ----
            if imageB.ndim == 3:
                grayB = rgb2gray(imageB)
            else:
                grayB = imageB.astype(np.float32)
            if grayB.max() > 1.0:
                grayB = grayB / 255.0

            grayB = exposure.equalize_adapthist(grayB)
            grayB_u8 = cv2.bilateralFilter((np.clip(grayB, 0, 1) * 255).astype(np.uint8), 9, 75, 75)

            expected_d_um = float(self.spin_crystal_d_um.value())
            min_sigma, max_sigma, num_sigma = self._sigma_range_from_expected_d_um(expected_d_um, pixel_to_um)
            blob_thresh = float(self.spin_crystal_thresh.value())

            blobs = blob_log(
                grayB_u8.astype(np.float32) / 255.0,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=blob_thresh
            )

            # Crystal size reference table
            if len(blobs) > 0:
                sigmas = blobs[:, 2]
                r_px = np.sqrt(2.0) * sigmas
                d_px = 2.0 * r_px
                d_um = d_px * pixel_to_um
                area_um2 = (np.pi * (r_px * pixel_to_um) ** 2)
                crystals_size_df = pd.DataFrame({
                    "Crystal_ID": np.arange(1, len(blobs) + 1),
                    "y_px": blobs[:, 0],
                    "x_px": blobs[:, 1],
                    "sigma_px": sigmas,
                    "diameter_um": d_um,
                    "area_um2": area_um2
                })
            else:
                crystals_size_df = pd.DataFrame(columns=["Crystal_ID","y_px","x_px","sigma_px","diameter_um","area_um2"])

            # Draw blobs as mask
            mask_B = np.zeros_like(grayB_u8, dtype=np.uint8)
            for y, x, r in blobs:
                rr, cc = np.ogrid[:mask_B.shape[0], :mask_B.shape[1]]
                circle = (rr - int(y))**2 + (cc - int(x))**2 <= int(r)**2
                mask_B[circle] = 255
            binary_B = mask_B

            # PL intensity histogram
            plt.figure(figsize=(8, 6))
            plt.hist(grayB_u8.ravel(), bins=256, range=[0, 255], alpha=0.7)
            plt.title('Histogram of Pixel Intensities (PL)')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
            hist_crystals_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_crystals.png")
            plt.savefig(hist_crystals_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            all_output_files.append(hist_crystals_image_path)

            # Resize for overlap
            filtered_binary_A_resized = cv2.resize((binary_A > 0).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)
            if binary_B.ndim == 3 and binary_B.shape[2] == 3:
                binary_B_gray = cv2.cvtColor(binary_B, cv2.COLOR_BGR2GRAY)
            else:
                binary_B_gray = binary_B
            binary_B_resized = cv2.resize((binary_B_gray > 0).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)

            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255

            # Mask bottom-right scalebar region (tweak if needed)
            h3, w3 = overlap.shape
            overlap[h3-80:h3, w3-1350:w3] = 0

            overlap_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Overlap.png")
            cv2.imwrite(overlap_path, overlap)
            all_output_files.append(overlap_path)

            # Mapping crystals to cells
            region_to_cell_mapping = []
            cell_labels = label(filtered_binary_A_resized)
            cell_props = regionprops(cell_labels)
            region_labels = label(overlap)
            region_props = regionprops(region_labels)
            cell_to_crystals = defaultdict(list)

            for region in region_props:
                region_coords = set(map(tuple, region.coords))
                best_match_cell = None
                max_overlap = 0
                for cell in cell_props:
                    cell_coords = set(map(tuple, cell.coords))
                    overlap_area = len(region_coords & cell_coords)
                    if overlap_area > max_overlap:
                        max_overlap = overlap_area
                        best_match_cell = cell.label
                region_to_cell_mapping.append({
                    "Region_Label": region.label,
                    "Associated_Cell": best_match_cell,
                    "Overlap (pixels)": max_overlap,
                    "Region_Area (pixels)": region.area,
                    "Region_Area (µm²)": region.area * (pixel_to_um ** 2)
                })
                if best_match_cell is not None:
                    cell_to_crystals[best_match_cell].append(region.label)

            df_mapping = pd.DataFrame(region_to_cell_mapping)
            if not df_mapping.empty and "Region_Area (µm²)" in df_mapping.columns:
                df_mapping = df_mapping[(df_mapping["Region_Area (µm²)"] > 0) & (df_mapping["Overlap (pixels)"] > 0)]
                df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())
                total_distinct_cells = df_mapping["Associated_Cell"].nunique()
                df_mapping["Total_Cells_with_crystals"] = total_distinct_cells
                total_area_cr = df_mapping["Region_Area (µm²)"].sum()
                total_row = ["", "", "", "Total Area Crystals", total_area_cr, "", ""]
                df_mapping.loc["Total"] = total_row
            else:
                total_distinct_cells = 0
                total_area_cr = 0.0

            cell_crystal_df = pd.DataFrame([
                {
                    "Cell_Label": cell_label,
                    "Crystal_Labels": ", ".join(map(str, crystals)),
                    "Crystal_Count": len(crystals)
                }
                for cell_label, crystals in cell_to_crystals.items()
            ])

            grouped_xlsx_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_All_Datasets.xlsx")
            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)
                df_mapping.to_excel(writer, sheet_name='Crystals', index=False)
                cell_crystal_df.to_excel(writer, sheet_name='Cell-to-crystal map', index=False)
                crystals_size_df.to_excel(writer, sheet_name='Crystal sizes', index=False)
            print(f"Saved results for {bf_file} to {grouped_xlsx_path}")

            # Visualization
            annotated_image = imageA.copy()
            for mapping in region_to_cell_mapping:
                region_label = mapping["Region_Label"]
                associated_cell = mapping["Associated_Cell"]
                if associated_cell:
                    region = next(r for r in region_props if r.label == region_label)
                    min_row, min_col, max_row, max_col = region.bbox
                    cv2.rectangle(annotated_image, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_image,
                        f"Cell {associated_cell}",
                        (min_col, min_row - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 0),
                        1
                    )

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(annotated_image, cmap='gray')
            ax[0].set_title('Detections')
            ax[0].axis('off')
            ax[1].imshow(overlap, cmap='gray')
            ax[1].set_title('Coincidences')
            ax[1].axis('off')
            plt.tight_layout()
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()

            annotated_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)
            all_output_files.append(annotated_image_path)

            del grayA, grayA_u8, binary_A, region_labels_A, region_props_A, overlay_image, filtered_binary_A
            del grayB, grayB_u8, binary_B, region_labels, region_props, overlap
            gc.collect()

            # Summary row (% of cells that contain at least one crystal)
            Percentage = f"{(total_distinct_cells / total_cells * 100):.2f}%" if total_cells > 0 else "0%"
            summary_rows.append({
                "Days": os.path.splitext(bf_file)[0],
                "total_cells": total_cells,
                "cells_with_crystals": total_distinct_cells,
                "%_cells_with_crystals": Percentage
            })

        # --------- Summary plot ----------
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df["Days"] = summary_df["Days"].astype(str)
            summary_df = summary_df.sort_values(by="Days")
            summary_df["%_cells_with_crystals"] = summary_df["%_cells_with_crystals"].astype(str).str.replace('%', '').astype(float)
            summary_df["Days"] = summary_df["Days"].str.extract(r"(\d+)")
            grouped_df = summary_df.groupby("Days").agg({
                "%_cells_with_crystals": ["mean", "std"]
            }).reset_index()
            grouped_df.columns = ["Days", "mean_percentage", "std_percentage"]
            grouped_df["Days"] = grouped_df["Days"].astype(int)
            grouped_df = grouped_df.sort_values(by="Days")
            max_percentage = grouped_df["mean_percentage"].max()
            y_max_limit = min(100, max_percentage + 10)

            plt.figure(figsize=(10, 6))
            plt.plot(grouped_df["Days"], grouped_df["mean_percentage"], marker='o', linestyle='-', linewidth=2, label="Average")
            for x, y, std in zip(grouped_df["Days"], grouped_df["mean_percentage"], grouped_df["std_percentage"]):
                plt.vlines(x=x, ymin=y - std, ymax=y + std, alpha=0.7, linewidth=2, label='±1 STD' if x == grouped_df["Days"].iloc[0] else "")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.title("Average % Cells with Crystals", fontsize=14)
            plt.xlabel("Days", fontsize=12)
            plt.ylabel("% Cells with Crystals", fontsize=12)
            plt.ylim(0, y_max_limit)
            plt.grid(True)
            plt.pause(0.001)
            QApplication.processEvents()
            plot_path = os.path.join(self.output_folder, "Plot.png")
            plt.savefig(plot_path, dpi=300)
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()

            grouped_df.to_excel(os.path.join(self.output_folder, "Plot.xlsx"), index=False)

        self.log("Processing complete!")

        # Zip outputs and remove individual images
        zip_path = os.path.join(self.output_folder, "All_Images_histograms.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in all_output_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))
        for file_path in all_output_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    # --------------- Pipeline 2: Areas -----------------
    def start_processing_2(self):
        self.processing_active = True
        self.stop_event.clear()

        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        try:
            pixel_to_um = self._compute_pixel_to_um()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for distance in pixels and known µm.")
            return None

        os.makedirs(self.output_folder, exist_ok=True)

        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.endswith('.tif')])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.endswith('.tif')])

        if len(bf_files) != len(pl_files):
            raise ValueError("Mismatch in the number of BF and PL .tif files.")

        all_output_files = []
        summary_rows = []
        min_cell_area_um2 = 20.0

        for bf_file, pl_file in zip(bf_files, pl_files):
            print(f"Processing: {bf_file} and {pl_file}")
            if self.stop_event.is_set():
                self.log("Processing stopped.")
                return

            self.log(f"Processing {bf_file} and {pl_file}...")

            bf_image_path = os.path.join(self.bf_folder, bf_file)
            pl_image_path = os.path.join(self.pl_folder, pl_file)
            imageA = cv2.imread(bf_image_path)
            imageB = cv2.imread(pl_image_path)

            if imageA is None or imageB is None:
                print(f"Skipping {bf_file} or {pl_file}: Unable to load image.")
                continue

            # ---- BF preprocessing & threshold (auto+ bias) ----
            grayA = rgb2gray(imageA)
            h, w = grayA.shape
            crop_margin_h = int(0.015 * h)
            crop_margin_w = int(0.025 * w)
            scale_mask = np.ones_like(grayA, dtype=bool)
            scale_mask[h - crop_margin_h:, w - crop_margin_w:] = False
            grayA = grayA * scale_mask
            grayA = exposure.equalize_adapthist(grayA)
            grayA_u8 = cv2.bilateralFilter((np.clip(grayA, 0, 1) * 255).astype(np.uint8), 9, 75, 75)

            bias = float(self.spin_thresh_bias.value())
            binary_A = self._auto_threshold_with_bias(grayA_u8, bias=bias)

            binary_A = morphology.remove_small_objects(binary_A.astype(bool), min_size=1600)
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=10000)
            binary_A = morphology.opening(binary_A)
            binary_A = (binary_A > 0).astype(np.uint8) * 255

            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            crop_start_row = h - crop_margin_h
            crop_start_col = w - crop_margin_w
            crop_mask = np.zeros_like(region_labels_A, dtype=bool)
            crop_mask[crop_start_row:, crop_start_col:] = True

            filtered_labels = []
            for region in region_props_A:
                region_mask = (region_labels_A == region.label)
                if not np.any(region_mask & crop_mask):
                    filtered_labels.append(region.label)

            new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
            label_counter = 1
            for lbl in filtered_labels:
                new_label_img[region_labels_A == lbl] = label_counter
                label_counter += 1

            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            areas = [region.area for region in region_props_A] or [1]
            media_area = np.median(areas)
            min_area = np.min(areas)
            std_area = np.std(areas)
            average = min_area + std_area

            # Save area histogram
            plt.figure(figsize=(8, 5))
            plt.hist(areas, bins=20, edgecolor='black')
            plt.title("Histogram of Region Areas")
            plt.xlabel("Area (pixels)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            hist_areas_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_Areas.png")
            plt.savefig(hist_areas_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            all_output_files.append(hist_areas_image_path)

            # Split with µm-driven min_distance
            for region in region_props_A:
                if region.area < average:
                    new_label_img[region.slice][region.image] = label_counter
                    label_counter += 1
                else:
                    region_mask = np.zeros_like(region_labels_A, dtype=np.uint8)
                    region_mask[region.slice][region.image] = 1
                    distance = ndi.distance_transform_edt(region_mask)
                    split_um = float(self.spin_split_um.value())
                    min_dist_px = max(1, int(round(split_um / pixel_to_um)))
                    coordinates = peak_local_max(distance, labels=region_mask, min_distance=min_dist_px)
                    local_maxi = np.zeros_like(distance, dtype=bool)
                    if coordinates.size > 0:
                        local_maxi[tuple(coordinates.T)] = True
                    markers = label(local_maxi)
                    labels_ws = watershed(-distance, markers, mask=region_mask)
                    for ws_label in np.unique(labels_ws):
                        if ws_label == 0:
                            continue
                        mask_ws = labels_ws == ws_label
                        new_label_img[mask_ws] = label_counter
                        label_counter += 1

            region_labels_A = new_label_img
            region_labels_A = label(region_labels_A > 0)
            region_props_A = regionprops(region_labels_A)

            if binary_A.shape != grayA_u8.shape:
                binary_A = resize(binary_A, grayA_u8.shape, order=0, preserve_range=True, anti_aliasing=False)

            seg_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented.png")
            cv2.imwrite(seg_path, region_labels_A.astype(np.uint16))
            all_output_files.append(seg_path)

            overlay_image = cv2.cvtColor((binary_A > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for region in regionprops(region_labels_A):
                y, x = region.centroid
                cv2.putText(overlay_image, str(region.label), (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            annotated_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented_Annotated.png")
            cv2.imwrite(annotated_path, overlay_image)
            all_output_files.append(annotated_path)

            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (
                        region_labels_A[min_row:max_row, min_col:max_col] == prop.label
                    )
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255

            region_area = pd.DataFrame({
                "Region_Label": [r.label for r in region_props_A],
                "Region_Area (pixels)": [r.area for r in region_props_A],
                "Region_Area (µm²)": [r.area * (pixel_to_um ** 2) for r in region_props_A]
            })

            region_area_df = region_area[region_area["Region_Area (µm²)"] > min_cell_area_um2]
            total_area = region_area_df["Region_Area (µm²)"].sum()
            total_cells = region_area_df["Region_Label"].count()

            region_area_df.loc["Total Area"] = ["", "Total Area", total_area]
            region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

            region_area_excel_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Region_Area_in_um2.xlsx")
            print(f"Saved region areas for {bf_file} to {region_area_excel_path}")

            plt.figure(figsize=(8, 6))
            plt.hist(grayA_u8.ravel(), bins=256, range=[0, 255], alpha=0.7)
            plt.title('Histogram of Pixel Intensities (BF)')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            all_output_files.append(hist_cells_image_path)

            # ---- PL blobs with parameterization ----
            if imageB.ndim == 3:
                grayB = rgb2gray(imageB)
            else:
                grayB = imageB.astype(np.float32)
            if grayB.max() > 1.0:
                grayB = grayB / 255.0
            grayB = exposure.equalize_adapthist(grayB)
            grayB_u8 = cv2.bilateralFilter((np.clip(grayB, 0, 1) * 255).astype(np.uint8), 9, 75, 75)

            expected_d_um = float(self.spin_crystal_d_um.value())
            min_sigma, max_sigma, num_sigma = self._sigma_range_from_expected_d_um(expected_d_um, pixel_to_um)
            blob_thresh = float(self.spin_crystal_thresh.value())

            blobs = blob_log(
                grayB_u8.astype(np.float32) / 255.0,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=blob_thresh
            )

            if len(blobs) > 0:
                sigmas = blobs[:, 2]
                r_px = np.sqrt(2.0) * sigmas
                d_px = 2.0 * r_px
                d_um = d_px * pixel_to_um
                area_um2 = (np.pi * (r_px * pixel_to_um) ** 2)
                crystals_size_df = pd.DataFrame({
                    "Crystal_ID": np.arange(1, len(blobs) + 1),
                    "y_px": blobs[:, 0],
                    "x_px": blobs[:, 1],
                    "sigma_px": sigmas,
                    "diameter_um": d_um,
                    "area_um2": area_um2
                })
            else:
                crystals_size_df = pd.DataFrame(columns=["Crystal_ID","y_px","x_px","sigma_px","diameter_um","area_um2"])

            mask_B = np.zeros_like(grayB_u8, dtype=np.uint8)
            for y, x, r in blobs:
                rr, cc = np.ogrid[:mask_B.shape[0], :mask_B.shape[1]]
                circle = (rr - int(y))**2 + (cc - int(x))**2 <= int(r)**2
                mask_B[circle] = 255
            binary_B = mask_B

            plt.figure(figsize=(8, 6))
            plt.hist(grayB_u8.ravel(), bins=256, range=[0, 255], alpha=0.7)
            plt.title('Histogram of Pixel Intensities (PL)')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
            hist_crystals_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_crystals.png")
            plt.savefig(hist_crystals_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            all_output_files.append(hist_crystals_image_path)

            filtered_binary_A_resized = cv2.resize((binary_A > 0).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)
            if binary_B.ndim == 3 and binary_B.shape[2] == 3:
                binary_B_gray = cv2.cvtColor(binary_B, cv2.COLOR_BGR2GRAY)
            else:
                binary_B_gray = binary_B
            binary_B_resized = cv2.resize((binary_B_gray > 0).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)

            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255
            h2, w2 = overlap.shape
            overlap[h2-80:h2, w2-1350:w2] = 0

            overlap_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Overlap.png")
            cv2.imwrite(overlap_path, overlap)
            all_output_files.append(overlap_path)

            region_to_cell_mapping = []
            cell_labels = label(filtered_binary_A_resized)
            cell_props = regionprops(cell_labels)
            region_labels = label(overlap)
            region_props = regionprops(region_labels)
            cell_to_crystals = defaultdict(list)

            for region in region_props:
                region_coords = set(map(tuple, region.coords))
                best_match_cell = None
                max_overlap = 0
                for cell in cell_props:
                    cell_coords = set(map(tuple, cell.coords))
                    overlap_area = len(region_coords & cell_coords)
                    if overlap_area > max_overlap:
                        max_overlap = overlap_area
                        best_match_cell = cell.label
                region_to_cell_mapping.append({
                    "Region_Label": region.label,
                    "Associated_Cell": best_match_cell,
                    "Overlap (pixels)": max_overlap,
                    "Region_Area (pixels)": region.area,
                    "Region_Area (µm²)": region.area * (pixel_to_um ** 2)
                })
                if best_match_cell is not None:
                    cell_to_crystals[best_match_cell].append(region.label)

            df_mapping = pd.DataFrame(region_to_cell_mapping)
            if not df_mapping.empty and "Region_Area (µm²)" in df_mapping.columns:
                df_mapping = df_mapping[(df_mapping["Region_Area (µm²)"] > 0) & (df_mapping["Overlap (pixels)"] > 0)]
                df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())
                total_distinct_cells = df_mapping["Associated_Cell"].nunique()
                df_mapping["Total_Cells_with_crystals"] = total_distinct_cells
                total_area_cr = df_mapping["Region_Area (µm²)"].sum()
                total_row = ["", "", "", "Total Area Crystals", total_area_cr, "", ""]
                df_mapping.loc["Total"] = total_row
            else:
                total_distinct_cells = 0
                total_area_cr = 0.0

            cell_crystal_df = pd.DataFrame([
                {
                    "Cell_Label": cell_label,
                    "Crystal_Labels": ", ".join(map(str, crystals)),
                    "Crystal_Count": len(crystals)
                }
                for cell_label, crystals in cell_to_crystals.items()
            ])

            grouped_xlsx_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_All_Datasets.xlsx")
            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)
                df_mapping.to_excel(writer, sheet_name='Crystals', index=False)
                cell_crystal_df.to_excel(writer, sheet_name='Cell-to-crystal map', index=False)
                crystals_size_df.to_excel(writer, sheet_name='Crystal sizes', index=False)
            print(f"Saved results for {bf_file} to {grouped_xlsx_path}")

            annotated_image = imageA.copy()
            for mapping in region_to_cell_mapping:
                region_label = mapping["Region_Label"]
                associated_cell = mapping["Associated_Cell"]
                if associated_cell:
                    region = next(r for r in region_props if r.label == region_label)
                    min_row, min_col, max_row, max_col = region.bbox
                    cv2.rectangle(annotated_image, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_image,
                        f"Cell {associated_cell}",
                        (min_col, min_row - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 0),
                        1
                    )

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(annotated_image, cmap='gray')
            ax[0].set_title('Detections')
            ax[0].axis('off')
            ax[1].imshow(overlap, cmap='gray')
            ax[1].set_title('Coincidences')
            ax[1].axis('off')
            plt.tight_layout()
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()

            annotated_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)
            all_output_files.append(annotated_image_path)

            del grayA, grayA_u8, binary_A, region_labels_A, region_props_A, overlay_image, filtered_binary_A
            del grayB, grayB_u8, binary_B, region_labels, region_props, overlap
            gc.collect()

            # Summary row: % area crystals / cells
            Percentage = f"{(total_area_cr / total_area * 100):.2f}%" if total_cells > 0 else "0%"
            summary_rows.append({
                "Days": os.path.splitext(bf_file)[0],
                "total_cells_area": total_area,
                "total_crystals_area": total_area_cr,
                "%_area_crystals_cells": Percentage
            })

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df["Days"] = summary_df["Days"].astype(str)
            summary_df = summary_df.sort_values(by="Days")
            summary_df["%_area_crystals_cells"] = summary_df["%_area_crystals_cells"].astype(str).str.replace('%', '').astype(float)
            summary_df["Days"] = summary_df["Days"].str.extract(r"(\d+)")

            grouped_df = summary_df.groupby("Days").agg({
                "%_area_crystals_cells": ["mean", "std"]
            }).reset_index()
            grouped_df.columns = ["Days", "mean_percentage", "std_percentage"]
            grouped_df["Days"] = grouped_df["Days"].astype(int)
            grouped_df = grouped_df.sort_values(by="Days")

            max_percentage = grouped_df["mean_percentage"].max()
            y_max_limit = min(100, max_percentage + 4)

            plt.figure(figsize=(10, 6))
            plt.plot(grouped_df["Days"], grouped_df["mean_percentage"], marker='o', linestyle='-', linewidth=2, label="Average")
            for x, y, std in zip(grouped_df["Days"], grouped_df["mean_percentage"], grouped_df["std_percentage"]):
                plt.vlines(x=x, ymin=y - std, ymax=y + std, alpha=0.7, linewidth=2, label='±1 STD' if x == grouped_df["Days"].iloc[0] else "")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.ylim(0, y_max_limit)
            plt.xlabel("Days")
            plt.ylabel("% Area Crystals / Cells")
            plt.title("Average % Area Crystals/Cells per Day")
            plt.grid(True)
            plt.pause(0.001)
            QApplication.processEvents()

            plot_path = os.path.join(self.output_folder, "Plot.png")
            plt.savefig(plot_path, dpi=300)
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()

            grouped_df.to_excel(os.path.join(self.output_folder, "Plot.xlsx"), index=False)

        self.log("Processing complete!")

        zip_path = os.path.join(self.output_folder, "All_Images_histograms.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in all_output_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))
        for file_path in all_output_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    # --------------- Pipeline 3: Number of cells -----------------
    def start_processing_3(self):
        self.processing_active = True
        self.stop_event.clear()

        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        try:
            pixel_to_um = self._compute_pixel_to_um()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for distance in pixels and known µm.")
            return None

        os.makedirs(self.output_folder, exist_ok=True)

        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.endswith('.tif')])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.endswith('.tif')])

        if len(bf_files) != len(pl_files):
            raise ValueError("Mismatch in the number of BF and PL .tif files.")

        all_output_files = []
        min_cell_area_um2 = 20.0

        for bf_file, pl_file in zip(bf_files, pl_files):
            print(f"Processing: {bf_file} and {pl_file}")
            if self.stop_event.is_set():
                self.log("Processing stopped.")
                return

            self.log(f"Processing {bf_file} and {pl_file}...")

            bf_image_path = os.path.join(self.bf_folder, bf_file)
            pl_image_path = os.path.join(self.pl_folder, pl_file)
            imageA = cv2.imread(bf_image_path)
            imageB = cv2.imread(pl_image_path)

            if imageA is None or imageB is None:
                print(f"Skipping {bf_file} or {pl_file}: Unable to load image.")
                continue

            # ---- BF preprocessing & threshold (auto + bias) ----
            grayA = rgb2gray(imageA)
            h, w = grayA.shape
            crop_margin_h = int(0.015 * h)
            crop_margin_w = int(0.025 * w)
            scale_mask = np.ones_like(grayA, dtype=bool)
            scale_mask[h - crop_margin_h:, w - crop_margin_w:] = False
            grayA = grayA * scale_mask
            grayA = exposure.equalize_adapthist(grayA)
            grayA_u8 = cv2.bilateralFilter((np.clip(grayA, 0, 1) * 255).astype(np.uint8), 9, 75, 75)

            bias = float(self.spin_thresh_bias.value())
            binary_A = self._auto_threshold_with_bias(grayA_u8, bias=bias)

            binary_A = morphology.remove_small_objects(binary_A.astype(bool), min_size=1600)
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=10000)
            binary_A = morphology.opening(binary_A)
            binary_A = (binary_A > 0).astype(np.uint8) * 255

            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            crop_start_row = h - crop_margin_h
            crop_start_col = w - crop_margin_w
            crop_mask = np.zeros_like(region_labels_A, dtype=bool)
            crop_mask[crop_start_row:, crop_start_col:] = True

            filtered_labels = []
            for region in region_props_A:
                region_mask = (region_labels_A == region.label)
                if not np.any(region_mask & crop_mask):
                    filtered_labels.append(region.label)

            new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
            label_counter = 1
            for lbl in filtered_labels:
                new_label_img[region_labels_A == lbl] = label_counter
                label_counter += 1

            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            areas = [region.area for region in region_props_A] or [1]
            media_area = np.median(areas)
            std_area = np.std(areas)
            min_area = np.min(areas)
            average = min_area  # keep logic as in original function

            # Save histogram of region areas
            plt.figure(figsize=(8, 5))
            plt.hist(areas, bins=20, edgecolor='black')
            plt.title("Histogram of Region Areas")
            plt.xlabel("Area (pixels)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            hist_areas_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_Areas.png")
            plt.savefig(hist_areas_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            all_output_files.append(hist_areas_image_path)

            # Split with µm-driven min_distance
            for region in region_props_A:
                if region.area < average:
                    new_label_img[region.slice][region.image] = label_counter
                    label_counter += 1
                else:
                    region_mask = np.zeros_like(region_labels_A, dtype=np.uint8)
                    region_mask[region.slice][region.image] = 1
                    distance = ndi.distance_transform_edt(region_mask)
                    split_um = float(self.spin_split_um.value())
                    min_dist_px = max(1, int(round(split_um / pixel_to_um)))
                    coordinates = peak_local_max(distance, labels=region_mask, min_distance=min_dist_px)
                    local_maxi = np.zeros_like(distance, dtype=bool)
                    if coordinates.size > 0:
                        local_maxi[tuple(coordinates.T)] = True
                    markers = label(local_maxi)
                    labels_ws = watershed(-distance, markers, mask=region_mask)
                    for ws_label in np.unique(labels_ws):
                        if ws_label == 0:
                            continue
                        mask_ws = labels_ws == ws_label
                        new_label_img[mask_ws] = label_counter
                        label_counter += 1

            region_labels_A = new_label_img
            region_labels_A = label(region_labels_A > 0)
            region_props_A = regionprops(region_labels_A)

            if binary_A.shape != grayA_u8.shape:
                binary_A = resize(binary_A, grayA_u8.shape, order=0, preserve_range=True, anti_aliasing=False)

            overlay_image = cv2.cvtColor((binary_A > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for region in regionprops(region_labels_A):
                y, x = region.centroid
                cv2.putText(overlay_image, str(region.label), (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            annotated_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented_Annotated.png")
            cv2.imwrite(annotated_path, overlay_image)
            all_output_files.append(annotated_path)

            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (
                        region_labels_A[min_row:max_row, min_col:max_col] == prop.label
                    )
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255

            region_area = pd.DataFrame({
                "Region_Label": [r.label for r in region_props_A],
                "Region_Area (pixels)": [r.area for r in region_props_A],
                "Region_Area (µm²)": [r.area * (pixel_to_um ** 2) for r in region_props_A]
            })

            region_area_df = region_area[region_area["Region_Area (µm²)"] > min_cell_area_um2]
            total_area = region_area_df["Region_Area (µm²)"].sum()
            total_cells = region_area_df["Region_Label"].count()

            region_area_df.loc["Total Area"] = ["", "Total Area", total_area]
            region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

            region_area_excel_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Region_Area_in_um2.xlsx")
            region_area_df.to_excel(region_area_excel_path, index=False)

            plt.figure(figsize=(8, 6))
            plt.hist(grayA_u8.ravel(), bins=256, range=[0, 255], alpha=0.7)
            plt.title('Histogram of Pixel Intensities (BF)')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            all_output_files.append(hist_cells_image_path)

            del grayA, grayA_u8, binary_A, region_labels_A, region_props_A, overlay_image, filtered_binary_A
            gc.collect()

        self.log("Processing complete!")

        zip_path = os.path.join(self.output_folder, "All_Images_histograms.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in all_output_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))
        for file_path in all_output_files:
            if os.path.exists(file_path):
                os.remove(file_path)

# Entry point of the application
if __name__ == "__main__":
    # Create a Qt application instance
    app = QApplication(sys.argv)

    # Instantiate the main window (custom image processing GUI)
    window = ImageProcessingApp()

    # Show the main window
    window.show()

    # Execute the Qt event loop and exit the application when it's closed
    sys.exit(app.exec_())


# In[ ]:




