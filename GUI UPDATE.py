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
    QTextEdit, QInputDialog, QComboBox
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

        # Label and input for pixel distance
        self.pixel_distance_label = QLabel("Distance in pixels:")
        self.pixel_distance_input = QLineEdit()
        self.pixel_distance_input.setText("NOT VALUE")

        # Label and combo box for known µm distances
        self.known_um_label = QLabel("Known distance (µm):")
        self.known_um_combo = QComboBox()
        self.known_um_combo.setEditable(True)
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

    def on_custom_um_entered(self):
        # Handle user entering a custom µm value in the combo box
        text = self.known_um_combo.currentText().strip()
    
        # If the entered text is not already in the combo box, add it
        if text not in [self.known_um_combo.itemText(i) for i in range(self.known_um_combo.count())]:
            self.known_um_combo.addItem(text)

    def update_pixel_distance(self):
        # Update the pixel distance input field based on the selected scale
        text = self.known_um_combo.currentText()
    
        # If the scale is known, set the corresponding px value; otherwise clear the field
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
            # Fallback to default values if file is missing
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
    
        # Only allow deletion of user-defined scales, not defaults
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
            # Warn if trying to delete a protected or non-existing scale
            QMessageBox.warning(self, "Not Found", f"The scale '{selected_scale}' can not be delete.")

    def start_processing(self):
        # Flag to indicate that processing is active
        self.processing_active = True

        # Reset the stop event in case it was triggered during a previous run
        self.stop_event.clear()

        # Validate that all necessary folders (BF, PL, and Output) have been selected
        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        try:
            # Read user input for scale calibration
            distance_in_px = float(self.pixel_distance_input.text())  # Distance in pixels (from scale bar)
            known_um = float(self.known_um_combo.currentText())       # Known real-world distance in micrometers

            # Prevent division by zero when calculating pixel-to-micron scale
            if distance_in_px == 0:
                raise ValueError("Distance in pixels cannot be zero.")
    
            # Compute pixel-to-micrometer conversion factor
            pixel_to_um = 1 / (known_um / distance_in_px)
        except ValueError:
            # Show warning if input is invalid or conversion fails
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for distance in pixels and known µm.")
            return None

        # Create the output directory if it doesn't already exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Collect and sort all .tif files in both BF and PL folders
        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.endswith('.tif')])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.endswith('.tif')])

        # Check that the number of BF and PL images match for paired processing
        if len(bf_files) != len(pl_files):
            raise ValueError("Mismatch in the number of BF and PL .tif files.")

        # List to keep track of output files generated during processing
        all_output_files = []

        # Placeholder for storing row data to summarize in Excel or logs
        summary_rows = []

        # Batch process each pair of Brightfield (BF) and Polarized Light (PL) images
        for bf_file, pl_file in zip(bf_files, pl_files):
            print(f"Processing: {bf_file} and {pl_file}")

            # Allow user to stop processing midway
            if self.stop_event.is_set():
                self.log("Processing stopped.")
                return

            self.log(f"Processing {bf_file} and {pl_file}...")

            # Load BF and PL images
            bf_image_path = os.path.join(self.bf_folder, bf_file)
            pl_image_path = os.path.join(self.pl_folder, pl_file)
            imageA = cv2.imread(bf_image_path)
            imageB = cv2.imread(pl_image_path)

            # Skip if images failed to load
            if imageA is None or imageB is None:
                print(f"Skipping {bf_file} or {pl_file}: Unable to load image.")
                continue

            # Convert BF image to grayscale
            grayA = rgb2gray(imageA)

            # --- Remove bottom-right scale bar region to avoid false detections ---
            h, w = grayA.shape
            crop_margin_h = int(0.015 * h)  # ~1.5% of height
            crop_margin_w = int(0.025 * w)  # ~2.5% of width
            
            # Mask the scale bar region (bottom-right) from analysis
            mask = np.ones_like(grayA, dtype=bool)
            mask[h - crop_margin_h:, w - crop_margin_w:] = False
            grayA = grayA * mask  # Apply mask to grayscale image

            # Enhance contrast using adaptive histogram equalization
            grayA = exposure.equalize_adapthist(grayA)

            # Denoise the image using bilateral filtering
            grayA = cv2.bilateralFilter((grayA * 255).astype(np.uint8), 9, 75, 75)

            # Segment the image using Otsu's thresholding
            threshold = threshold_otsu(grayA)
            binary_A = (grayA < threshold).astype(np.uint8) * 255

            # Apply morphological operations to clean segmentation
            binary_A = morphology.remove_small_objects(binary_A, min_size=1600)#1000
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=10000)#4000
            binary_A = morphology.opening(binary_A)
            
            binary_A = (binary_A > 0).astype(np.uint8) * 255
            
            # Label connected regions
            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            # Create mask for excluding cropped scale bar area
            crop_start_row = h - crop_margin_h
            crop_start_col = w - crop_margin_w
            crop_mask = np.zeros_like(region_labels_A, dtype=bool)
            crop_mask[crop_start_row:, crop_start_col:] = True

            # Filter out regions that intersect with the cropped area
            filtered_labels = []
            for region in region_props_A:
                region_mask = (region_labels_A == region.label)
                if not np.any(region_mask & crop_mask):
                    filtered_labels.append(region.label)

            # Generate new label image without excluded regions
            new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
            label_counter = 1
            for lbl in filtered_labels:
                new_label_img[region_labels_A == lbl] = label_counter
                label_counter += 1

            # Refresh region labels and properties
            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            # Calculate region area statistics for filtering/splitting
            areas = [region.area for region in region_props_A]
            media_area = np.median(areas)
            min_area = np.min(areas)
            std_area = np.std(areas)
            average = min_area + std_area  # Adaptive threshold

            # --- Save histogram of region areas ---
            plt.figure(figsize=(8, 5))
            plt.hist(areas, bins=20, color='skyblue', edgecolor='black')
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

            print(f"Saved histogram for {bf_file} to {hist_areas_image_path}")
            all_output_files.append(hist_areas_image_path)

            # Refine label image: keep small regions, split large ones using watershed
            for region in region_props_A:
                if region.area < average:
                    new_label_img[region.slice][region.image] = label_counter
                    label_counter += 1
                else:
                    region_mask = np.zeros_like(region_labels_A, dtype=np.uint8)
                    region_mask[region.slice][region.image] = 1
                    distance = ndi.distance_transform_edt(region_mask)
                    coordinates = peak_local_max(distance, labels=region_mask, min_distance=5)
                    local_maxi = np.zeros_like(distance, dtype=bool)
                    local_maxi[tuple(coordinates.T)] = True
                    markers = label(local_maxi)
                    labels_ws = watershed(-distance, markers, mask=region_mask)
                    for ws_label in np.unique(labels_ws):
                        if ws_label == 0:
                            continue
                        mask = labels_ws == ws_label
                        new_label_img[mask] = label_counter
                        label_counter += 1

            # Final labeled image after splitting
            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)
           
            # 🔥 Reset labels to start from 1
            region_labels_A = label(region_labels_A > 0)
            region_props_A = regionprops(region_labels_A)
            
            # Ensure binary mask matches grayscale shape
            if binary_A.shape != grayA.shape:
                binary_A = resize(binary_A, grayA.shape, order=0, preserve_range=True, anti_aliasing=False)

            # --- Visualize segmentation ---
            plt.figure(figsize=(8, 8))
            plt.imshow(region_labels_A, cmap='viridis')
            plt.title('Segmentation')
            plt.axis('off')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()

            # Save binary_A
            seg_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented.png")
            cv2.imwrite(seg_path, region_labels_A)
            print(f"Saved segmented image to {seg_path}")
            all_output_files.append(seg_path)

            # Annotate region labels on binary image
            overlay_image = cv2.cvtColor((binary_A > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for region in regionprops(region_labels_A):
                y, x = region.centroid
                label_id = region.label
                cv2.putText(overlay_image, str(region.label), (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Save annotated segmentation image
            annotated_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented_Annotated.png")
            cv2.imwrite(annotated_path, overlay_image)
            print(f"Saved annotated image with labels to {annotated_path}")
            all_output_files.append(annotated_path)

            # Create binary mask with only valid detected regions
            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (
                        region_labels_A[min_row:max_row, min_col:max_col] == prop.label
                    )
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255
                      # --- Save region statistics to Excel ---
            region_area = pd.DataFrame({
                "Region_Label": [region.label for region in region_props_A],
                "Region_Area (pixels)": [region.area for region in region_props_A],
                "Region_Area (µm²)": [region.area * (pixel_to_um ** 2) for region in region_props_A]
            })

            # Filter out regions with non-positive area (shouldn't happen, but safe check)
            region_area_df = region_area[region_area["Region_Area (pixels)"] > 800]
            
            total_area = region_area_df["Region_Area (µm²)"].sum()
            total_cells = region_area_df["Region_Label"].count()

            # Append summary rows
            region_area_df.loc["Total Area"] = ["", "Total Area", total_area]
            region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

            # Save region stats to Excel
            region_area_excel_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Region_Area_in_um2.xlsx")
            print(f"Saved region areas for {bf_file} to {region_area_excel_path}")

            # --- Plot histogram of pixel intensities ---
            plt.figure(figsize=(8, 6))
            plt.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()

            # Save the pixel intensity histogram
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            print(f"Saved histogram for {bf_file} to {annotated_path}")
            all_output_files.append(hist_cells_image_path)

            # grayB should be float [0-1] for blob_log
            if imageB.ndim == 3:
                grayB = rgb2gray(imageB)
            if grayB.max() > 1.0:
                grayB = grayB / 255.0

            grayB = exposure.equalize_adapthist(grayB)

            # Apply bilateral filter to reduce noise
            grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)

            blobs = blob_log(grayB, min_sigma=1, max_sigma=8, num_sigma=3, threshold=0.02)  

            # Make a blank mask
            mask_B = np.zeros_like(grayB, dtype=np.uint8)

            # Draw blobs as filled circles
            for y, x, r in blobs:
                rr, cc = np.ogrid[:mask_B.shape[0], :mask_B.shape[1]]
                circle = (rr - int(y))**2 + (cc - int(x))**2 <= int(r)**2
                mask_B[circle] = 255

            binary_B = mask_B
            
            plt.figure(figsize=(8, 6))
            plt.hist(grayB.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
    
            # Save the histogram image
            hist_crystals_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_crystals.png")
            plt.savefig(hist_crystals_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            plt.close()
            print(f"Saved histogram for {bf_file} to {hist_crystals_image_path}")
            all_output_files.append(hist_crystals_image_path)
    
            QApplication.processEvents()  # Refresh PyQt GUI
    
            # Resize for alignment
            filtered_binary_A_resized = cv2.resize((binary_A > 0).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)
            
            # If binary_B is a 3-channel (color) image, convert to grayscale
            if binary_B.ndim == 3 and binary_B.shape[2] == 3:
                binary_B_gray = cv2.cvtColor(binary_B, cv2.COLOR_BGR2GRAY)
            else:
                binary_B_gray = binary_B

            binary_B_resized = cv2.resize((binary_B_gray > 0).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)

            # Overlap calculation
            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255

            # 🔽 Mask the scale bar in bottom-right (adjust size as needed)
            h3, w3 = overlap.shape
            overlap[h3-80:h3, w3-1350:w3] = 0  # adjust 50 and 100 depending on the size of the scale bar#aumentar el 300 y un poco el 5#450 ORIGINAL
    
            # Save overlap results
            overlap_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Overlap.png")
            cv2.imwrite(overlap_path, overlap)
            all_output_files.append(overlap_path)

            # Save clustering information
            region_to_cell_mapping = []
            cell_labels = label(filtered_binary_A_resized)
            cell_props = regionprops(cell_labels)
            region_labels = label(overlap)
            region_props = regionprops(region_labels)

            cell_to_crystals = defaultdict(list)

            for region in region_props:
                region_coords = set(tuple(coord) for coord in region.coords)
                best_match_cell = None
                max_overlap = 0
                for cell in cell_props:
                    cell_coords = set(tuple(coord) for coord in cell.coords)
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

                # ✅ Store the crystal label for the matched cell
                if best_match_cell is not None:
                    cell_to_crystals[best_match_cell].append(region.label)

            # Save region-to-cell mapping as CSV
            df_mapping = pd.DataFrame(region_to_cell_mapping)

            #df_mapping = df_mapp[df_mapp["Region_Area (pixels)"] > 0]

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
            
            # Save cell-to-crystal list (for debugging or export) ---
            cell_crystal_df = pd.DataFrame([
                {
                    "Cell_Label": cell_label,
                    "Crystal_Labels": ", ".join(map(str, crystals)),
                    "Crystal_Count": len(crystals)
                }
                for cell_label, crystals in cell_to_crystals.items()
            ])

            # --- Save Excel ---
            mapping_excel_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Region_Cell_Mapping.xlsx")

            grouped_xlsx_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_All_Datasets.xlsx")

            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)
                df_mapping.to_excel(writer, sheet_name='Crystals', index=False)
                cell_crystal_df.to_excel(writer, sheet_name='Cell-to-crystal map', index=False)
            
            print(f"Saved results for {bf_file} to {grouped_xlsx_path}")
            #--------------------------------------------------------------
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
            
            # Plot both binary_A and binary_B
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Show detections
            ax[0].imshow(annotated_image, cmap='gray')
            ax[0].set_title('Detections')
            ax[0].axis('off')  # Hide axes

            # Show coincidences
            ax[1].imshow(overlap, cmap='gray')
            ax[1].set_title('Coincidences')
            ax[1].axis('off')  # Hide axes

            plt.tight_layout()
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            plt.close()
    
            # Save annotated image
            annotated_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)

            print(f"Saved results for {bf_file} to {self.output_folder}")    
            
            all_output_files.append(annotated_image_path)

            del grayA, binary_A, region_labels_A, region_props_A, overlay_image, filtered_binary_A, grayB, binary_B, region_labels, region_props, overlap
            gc.collect()

            # ------------------- Summary -------------------

            # Calculate the percentage of cells that contain at least one crystal
            Percentage = f"{(total_distinct_cells / total_cells * 100):.2f}%" if total_cells > 0 else "0%"

            # Append summary statistics for this image to the report
            summary_rows.append({
                "Days": os.path.splitext(bf_file)[0],      # Use base filename (without extension) as day identifier
                "total_cells": total_cells,               # Total number of segmented cell regions
                "cells_with_crystals": total_distinct_cells,  # Number of cells that contain one or more crystals
                "%_cells_with_crystals": Percentage       # Percent of cells with crystals (formatted as string with %)
            })

        # -------------------- Generate Summary Plot --------------------

        # Create a DataFrame from the collected summary information
        summary_df = pd.DataFrame(summary_rows)

        # Ensure the "Day" column is treated as a string for proper sorting
        summary_df["Days"] = summary_df["Days"].astype(str)
        summary_df = summary_df.sort_values(by="Days")

        # Convert percentage column from string (e.g., "12.5%") to float (e.g., 12.5)
        summary_df["%_cells_with_crystals"] = summary_df["%_cells_with_crystals"].astype(str).str.replace('%', '').astype(float)

        # Extract numeric part from the "Day" string for grouping (e.g., "3A" → 3)
        summary_df["Days"] = summary_df["Days"].str.extract(r"(\d+)")  # Only digits

        # Group by numeric day and compute mean and standard deviation of the percentages
        grouped_df = summary_df.groupby("Days").agg({
            "%_cells_with_crystals": ["mean", "std"]
        }).reset_index()

        # Flatten multi-level column names after aggregation
        grouped_df.columns = ["Days", "mean_percentage", "std_percentage"]

        # Convert DAYS to integer for proper numerical sorting
        grouped_df["Days"] = grouped_df["Days"].astype(int)
        grouped_df = grouped_df.sort_values(by="Days")

        # Determine the Y-axis limit (max percentage + buffer, capped at 100%)
        max_percentage = grouped_df["mean_percentage"].max()
        y_max_limit = min(100, max_percentage + 10)

        # Plot average % of cells with crystals per day
        plt.figure(figsize=(10, 6))
        plt.plot(
            grouped_df["Days"],
            grouped_df["mean_percentage"],
            marker='o',
            linestyle='-',
            color='blue',
            linewidth=2,
            label="Average"
        )

        # Draw vertical lines for ±1 standard deviation
        for x, y, std in zip(grouped_df["Days"], grouped_df["mean_percentage"], grouped_df["std_percentage"]):
            plt.vlines(
                x=x,
                ymin=y - std,
                ymax=y + std,
                color='blue',
                alpha=0.7,
                linewidth=2,
                label='±1 STD' if x == grouped_df["Days"].iloc[0] else ""
            )

        # Avoid duplicate legend entries
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

        # Save the plot image
        plot_path = os.path.join(self.output_folder, "Plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.pause(0.001)
        QApplication.processEvents()
        plt.close()

        # Save the grouped summary data to Excel
        grouped_df.to_excel(os.path.join(self.output_folder, "Plot.xlsx"), index=False)

        self.log("Processing complete!")

        # -------------------- Zip Result Files --------------------
        # Create a ZIP archive with all saved result images
        zip_path = os.path.join(self.output_folder, "All_Images_histograms.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in all_output_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))

        # Delete the individual files after zipping
        for file_path in all_output_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    def start_processing_2(self):
        # Flag to indicate that processing is active
        self.processing_active = True

        # Reset the stop event in case it was triggered during a previous run
        self.stop_event.clear()

        # Validate that all necessary folders (BF, PL, and Output) have been selected
        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        try:
            # Read user input for scale calibration
            distance_in_px = float(self.pixel_distance_input.text())  # Distance in pixels (from scale bar)
            known_um = float(self.known_um_combo.currentText())       # Known real-world distance in micrometers

            # Prevent division by zero when calculating pixel-to-micron scale
            if distance_in_px == 0:
                raise ValueError("Distance in pixels cannot be zero.")
    
            # Compute pixel-to-micrometer conversion factor
            pixel_to_um = 1 / (known_um / distance_in_px)
        except ValueError:
            # Show warning if input is invalid or conversion fails
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for distance in pixels and known µm.")
            return None

        # Create the output directory if it doesn't already exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Collect and sort all .tif files in both BF and PL folders
        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.endswith('.tif')])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.endswith('.tif')])

        # Check that the number of BF and PL images match for paired processing
        if len(bf_files) != len(pl_files):
            raise ValueError("Mismatch in the number of BF and PL .tif files.")

        # List to keep track of output files generated during processing
        all_output_files = []

        # Placeholder for storing row data to summarize in Excel or logs
        summary_rows = []

        # Batch process each pair of Brightfield (BF) and Polarized Light (PL) images
        for bf_file, pl_file in zip(bf_files, pl_files):
            print(f"Processing: {bf_file} and {pl_file}")

            # Allow user to stop processing midway
            if self.stop_event.is_set():
                self.log("Processing stopped.")
                return

            self.log(f"Processing {bf_file} and {pl_file}...")

            # Load BF and PL images
            bf_image_path = os.path.join(self.bf_folder, bf_file)
            pl_image_path = os.path.join(self.pl_folder, pl_file)
            imageA = cv2.imread(bf_image_path)
            imageB = cv2.imread(pl_image_path)

            # Skip if images failed to load
            if imageA is None or imageB is None:
                print(f"Skipping {bf_file} or {pl_file}: Unable to load image.")
                continue

            # Convert BF image to grayscale
            grayA = rgb2gray(imageA)

            # --- Remove bottom-right scale bar region to avoid false detections ---
            h, w = grayA.shape
            crop_margin_h = int(0.015 * h)  # ~1.5% of height
            crop_margin_w = int(0.025 * w)  # ~2.5% of width
            
            # Mask the scale bar region (bottom-right) from analysis
            mask = np.ones_like(grayA, dtype=bool)
            mask[h - crop_margin_h:, w - crop_margin_w:] = False
            grayA = grayA * mask  # Apply mask to grayscale image

            # Enhance contrast using adaptive histogram equalization
            grayA = exposure.equalize_adapthist(grayA)

            # Denoise the image using bilateral filtering
            grayA = cv2.bilateralFilter((grayA * 255).astype(np.uint8), 9, 75, 75)

            # Segment the image using Otsu's thresholding
            threshold = threshold_otsu(grayA)
            binary_A = (grayA < threshold).astype(np.uint8) * 255

            # Apply morphological operations to clean segmentation
            binary_A = morphology.remove_small_objects(binary_A, min_size=1600)#1000
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=10000)#4000
            binary_A = morphology.opening(binary_A)
      
            binary_A = (binary_A > 0).astype(np.uint8) * 255
            
            # Label connected regions
            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            # Create mask for excluding cropped scale bar area
            crop_start_row = h - crop_margin_h
            crop_start_col = w - crop_margin_w
            crop_mask = np.zeros_like(region_labels_A, dtype=bool)
            crop_mask[crop_start_row:, crop_start_col:] = True

            # Filter out regions that intersect with the cropped area
            filtered_labels = []
            for region in region_props_A:
                region_mask = (region_labels_A == region.label)
                if not np.any(region_mask & crop_mask):
                    filtered_labels.append(region.label)

            # Generate new label image without excluded regions
            new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
            label_counter = 1
            for lbl in filtered_labels:
                new_label_img[region_labels_A == lbl] = label_counter
                label_counter += 1

            # Refresh region labels and properties
            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            # Calculate region area statistics for filtering/splitting
            areas = [region.area for region in region_props_A]
            media_area = np.median(areas)
            min_area = np.min(areas)
            std_area = np.std(areas)
            average = min_area + std_area  # Adaptive threshold

            # --- Save histogram of region areas ---
            plt.figure(figsize=(8, 5))
            plt.hist(areas, bins=20, color='skyblue', edgecolor='black')
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
            print(f"Saved histogram for {bf_file} to {hist_areas_image_path}")
            all_output_files.append(hist_areas_image_path)

            # Refine label image: keep small regions, split large ones using watershed
            for region in region_props_A:
                if region.area < average:
                    new_label_img[region.slice][region.image] = label_counter
                    label_counter += 1
                else:
                    region_mask = np.zeros_like(region_labels_A, dtype=np.uint8)
                    region_mask[region.slice][region.image] = 1
                    distance = ndi.distance_transform_edt(region_mask)
                    coordinates = peak_local_max(distance, labels=region_mask, min_distance=5)
                    local_maxi = np.zeros_like(distance, dtype=bool)
                    local_maxi[tuple(coordinates.T)] = True
                    markers = label(local_maxi)
                    labels_ws = watershed(-distance, markers, mask=region_mask)
                    for ws_label in np.unique(labels_ws):
                        if ws_label == 0:
                            continue
                        mask = labels_ws == ws_label
                        new_label_img[mask] = label_counter
                        label_counter += 1

            # Final labeled image after splitting
            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            # 🔥 Reset labels to start from 1
            region_labels_A = label(region_labels_A > 0)
            region_props_A = regionprops(region_labels_A)

            # Ensure binary mask matches grayscale shape
            if binary_A.shape != grayA.shape:
                binary_A = resize(binary_A, grayA.shape, order=0, preserve_range=True, anti_aliasing=False)

            # --- Visualize segmentation ---
            plt.figure(figsize=(8, 8))
            plt.imshow(region_labels_A, cmap='viridis')
            plt.title('Segmentation')
            plt.axis('off')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()

            # Save binary_A
            seg_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented.png")
            cv2.imwrite(seg_path, region_labels_A)
            print(f"Saved segmented image to {seg_path}")
            all_output_files.append(seg_path)

            # Annotate region labels on binary image
            overlay_image = cv2.cvtColor((binary_A > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for region in regionprops(region_labels_A):
                y, x = region.centroid
                label_id = region.label
                cv2.putText(overlay_image, str(region.label), (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Save annotated segmentation image
            annotated_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented_Annotated.png")
            cv2.imwrite(annotated_path, overlay_image)
            print(f"Saved annotated image with labels to {annotated_path}")
            all_output_files.append(annotated_path)

            # Create binary mask with only valid detected regions
            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (
                        region_labels_A[min_row:max_row, min_col:max_col] == prop.label
                    )
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255

            # --- Save region statistics to Excel ---
            region_area = pd.DataFrame({
                "Region_Label": [region.label for region in region_props_A],
                "Region_Area (pixels)": [region.area for region in region_props_A],
                "Region_Area (µm²)": [region.area * (pixel_to_um ** 2) for region in region_props_A]
            })

            # Filter out regions with non-positive area (shouldn't happen, but safe check)
            region_area_df = region_area[region_area["Region_Area (pixels)"] > 800]
            
            total_area = region_area_df["Region_Area (µm²)"].sum()
            total_cells = region_area_df["Region_Label"].count()

            # Append summary rows
            region_area_df.loc["Total Area"] = ["", "Total Area", total_area]
            region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

            # Save region stats to Excel
            region_area_excel_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Region_Area_in_um2.xlsx")
            print(f"Saved region areas for {bf_file} to {region_area_excel_path}")

            # --- Plot histogram of pixel intensities ---
            plt.figure(figsize=(8, 6))
            plt.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()

            # Save the pixel intensity histogram
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            print(f"Saved histogram for {bf_file} to {annotated_path}")
            all_output_files.append(hist_cells_image_path)
            
            # grayB should be float [0-1] for blob_log
            if imageB.ndim == 3:
                grayB = rgb2gray(imageB)
            if grayB.max() > 1.0:
                grayB = grayB / 255.0

            grayB = exposure.equalize_adapthist(grayB)

            # Apply bilateral filter to reduce noise
            grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)

            blobs = blob_log(grayB, min_sigma=1, max_sigma=8, num_sigma=3, threshold=0.02)  

            # Make a blank mask
            mask_B = np.zeros_like(grayB, dtype=np.uint8)

            # Draw blobs as filled circles
            for y, x, r in blobs:
                rr, cc = np.ogrid[:mask_B.shape[0], :mask_B.shape[1]]
                circle = (rr - int(y))**2 + (cc - int(x))**2 <= int(r)**2
                mask_B[circle] = 255

            binary_B = mask_B
            
            plt.figure(figsize=(8, 6))
            plt.hist(grayB.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
    
            # Save the histogram image
            hist_crystals_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_crystals.png")
            plt.savefig(hist_crystals_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            plt.close()
            print(f"Saved histogram for {bf_file} to {hist_crystals_image_path}")
            all_output_files.append(hist_crystals_image_path)
    
            QApplication.processEvents()  # Refresh PyQt GUI
    
            # Resize for alignment
            filtered_binary_A_resized = cv2.resize((binary_A > 0).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)
            
            # If binary_B is a 3-channel (color) image, convert to grayscale
            if binary_B.ndim == 3 and binary_B.shape[2] == 3:
                binary_B_gray = cv2.cvtColor(binary_B, cv2.COLOR_BGR2GRAY)
            else:
                binary_B_gray = binary_B

            binary_B_resized = cv2.resize((binary_B_gray > 0).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)

            # Overlap calculation
            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255

            # 🔽 Mask the scale bar in bottom-right (adjust size as needed)
            h2, w2 = overlap.shape
            overlap[h2-80:h2, w2-1350:w2] = 0  # adjust 50 and 100 depending on the size of the scale bar#aumentar el 300 y un poco el 50
    
            # Save overlap results
            overlap_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Overlap.png")
            cv2.imwrite(overlap_path, overlap)
            all_output_files.append(overlap_path)

            # Save clustering information
            region_to_cell_mapping = []
            cell_labels = label(filtered_binary_A_resized)
            cell_props = regionprops(cell_labels)
            region_labels = label(overlap)
            region_props = regionprops(region_labels)
            cell_to_crystals = defaultdict(list)
            
            for region in region_props:
                region_coords = set(tuple(coord) for coord in region.coords)
                best_match_cell = None
                max_overlap = 0
                for cell in cell_props:
                    cell_coords = set(tuple(coord) for coord in cell.coords)
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

                # ✅ Store the crystal label for the matched cell
                if best_match_cell is not None:
                    cell_to_crystals[best_match_cell].append(region.label)

            # Save region-to-cell mapping as CSV
            df_mapping = pd.DataFrame(region_to_cell_mapping)

            #df_mapping = df_mapp[df_mapp["Region_Area (pixels)"] > 0]

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
            
            # --- Optional: Save cell-to-crystal list (for debugging or export) ---
            cell_crystal_df = pd.DataFrame([
                {
                    "Cell_Label": cell_label,
                    "Crystal_Labels": ", ".join(map(str, crystals)),
                    "Crystal_Count": len(crystals)
                }
                for cell_label, crystals in cell_to_crystals.items()
            ])

            # --- Save Excel ---
            mapping_excel_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Region_Cell_Mapping.xlsx")

            grouped_xlsx_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_All_Datasets.xlsx")

            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)
                df_mapping.to_excel(writer, sheet_name='Crystals', index=False)
                cell_crystal_df.to_excel(writer, sheet_name='Cell-to-crystal map', index=False)
            
            print(f"Saved results for {bf_file} to {grouped_xlsx_path}")
            #--------------------------------------------------------------
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
            
            # Plot both binary_A and binary_B
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Show detections
            ax[0].imshow(annotated_image, cmap='gray')
            ax[0].set_title('Detections')
            ax[0].axis('off')  # Hide axes

            # Show coincidences
            ax[1].imshow(overlap, cmap='gray')
            ax[1].set_title('Coincidences')
            ax[1].axis('off')  # Hide axes

            plt.tight_layout()
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            plt.close()
    
            # Save annotated image
            annotated_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)

            print(f"Saved results for {bf_file} to {self.output_folder}")    
            
            all_output_files.append(annotated_image_path)

            del grayA, binary_A, region_labels_A, region_props_A, overlay_image, filtered_binary_A, grayB, binary_B, region_labels, region_props, overlap
            gc.collect()

            # Calculate the percentage of crystal-covered area relative to total cell area
            Percentage = f"{(total_area_cr / total_area * 100):.2f}%" if total_cells > 0 else "0%"

            # Append summary information for this image to the report
            summary_rows.append({
                "Days": os.path.splitext(bf_file)[0],            # Extract image identifier from filename
                "total_cells_area": total_area,                 # Sum of all cell region areas in µm²
                "total_crystals_area": total_area_cr,           # Sum of all crystal region areas in µm²
                "%_area_crystals_cells": Percentage             # Area percentage of crystals relative to cells
            })

        # Create a DataFrame from all summarized results
        summary_df = pd.DataFrame(summary_rows)

        # Ensure 'Day' is treated as a string for consistent sorting
        summary_df["Days"] = summary_df["Days"].astype(str)
        summary_df = summary_df.sort_values(by="Days")

        # Convert percentage string to float if needed (e.g., "23.5%" → 23.5)
        summary_df["%_area_crystals_cells"] = summary_df["%_area_crystals_cells"].astype(str).str.replace('%', '').astype(float)

        # Extract numeric portion of the day (e.g., "1A" → 1) to group by day
        summary_df["Days"] = summary_df["Days"].str.extract(r"(\d+)")  # Extract digits only

        # Group by day number and compute mean and standard deviation of percentage
        grouped_df = summary_df.groupby("Days").agg({
            "%_area_crystals_cells": ["mean", "std"]
        }).reset_index()

        # Flatten multi-index column names
        grouped_df.columns = ["Days", "mean_percentage", "std_percentage"]

        # Convert DAYS to integer and sort numerically
        grouped_df["Days"] = grouped_df["Days"].astype(int)
        grouped_df = grouped_df.sort_values(by="Days")

        # Determine Y-axis limit for the plot
        max_percentage = grouped_df["mean_percentage"].max()
        y_max_limit = min(100, max_percentage + 4)  # Cap at 100%

        # Plot average % of cells with crystals per day
        plt.figure(figsize=(10, 6))
        plt.plot(
            grouped_df["Days"],
            grouped_df["mean_percentage"],
            marker='o',
            linestyle='-',
            color='blue',
            linewidth=2,
            label="Average"
        )

        # Draw vertical lines for ±1 standard deviation
        for x, y, std in zip(grouped_df["Days"], grouped_df["mean_percentage"], grouped_df["std_percentage"]):
            plt.vlines(
                x=x,
                ymin=y - std,
                ymax=y + std,
                color='blue',
                alpha=0.7,
                linewidth=2,
                label='±1 STD' if x == grouped_df["Days"].iloc[0] else ""
            )

        # Avoid duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.ylim(0, y_max_limit)
        plt.xlabel("Days")
        plt.ylabel("% Area Crystals / Cells")
        plt.title("Average % Area Crystals/Cells per Day")
        plt.grid(True)
        plt.pause(0.001)
        QApplication.processEvents()  # Update PyQt GUI

        # Save the plot as PNG
        plot_path = os.path.join(self.output_folder, "Plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.pause(0.001)
        QApplication.processEvents()
        plt.close()

        # Export grouped summary data to Excel
        grouped_df.to_excel(os.path.join(self.output_folder, "Plot.xlsx"), index=False)

        self.log("Processing complete!")

        # -----------------------------------------------------
        # Create a ZIP archive with all output histogram and annotated image files
        zip_path = os.path.join(self.output_folder, "All_Images_histograms.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in all_output_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))

        # Remove the original files after archiving
        for file_path in all_output_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                
    def start_processing_3(self):
        # Flag to indicate that processing is active
        self.processing_active = True

        # Reset the stop event in case it was triggered during a previous run
        self.stop_event.clear()

        # Validate that all necessary folders (BF, PL, and Output) have been selected
        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        try:
            # Read user input for scale calibration
            distance_in_px = float(self.pixel_distance_input.text())  # Distance in pixels (from scale bar)
            known_um = float(self.known_um_combo.currentText())       # Known real-world distance in micrometers

            # Prevent division by zero when calculating pixel-to-micron scale
            if distance_in_px == 0:
                raise ValueError("Distance in pixels cannot be zero.")
    
            # Compute pixel-to-micrometer conversion factor
            pixel_to_um = 1 / (known_um / distance_in_px)
        except ValueError:
            # Show warning if input is invalid or conversion fails
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for distance in pixels and known µm.")
            return None

        # Create the output directory if it doesn't already exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Collect and sort all .tif files in both BF and PL folders
        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.endswith('.tif')])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.endswith('.tif')])

        # Check that the number of BF and PL images match for paired processing
        if len(bf_files) != len(pl_files):
            raise ValueError("Mismatch in the number of BF and PL .tif files.")

        # List to keep track of output files generated during processing
        all_output_files = []

        # Batch process each pair of Brightfield (BF) and Polarized Light (PL) images
        for bf_file, pl_file in zip(bf_files, pl_files):
            print(f"Processing: {bf_file} and {pl_file}")

            # Allow user to stop processing midway
            if self.stop_event.is_set():
                self.log("Processing stopped.")
                return

            self.log(f"Processing {bf_file} and {pl_file}...")

            # Load BF and PL images
            bf_image_path = os.path.join(self.bf_folder, bf_file)
            pl_image_path = os.path.join(self.pl_folder, pl_file)
            imageA = cv2.imread(bf_image_path)
            imageB = cv2.imread(pl_image_path)

            # Skip if images failed to load
            if imageA is None or imageB is None:
                print(f"Skipping {bf_file} or {pl_file}: Unable to load image.")
                continue

            # Convert BF image to grayscale
            grayA = rgb2gray(imageA)

            # --- Remove bottom-right scale bar region to avoid false detections ---
            h, w = grayA.shape
            crop_margin_h = int(0.015 * h)  # ~1.5% of height
            crop_margin_w = int(0.025 * w)  # ~2.5% of width
            
            # Mask the scale bar region (bottom-right) from analysis
            mask = np.ones_like(grayA, dtype=bool)
            mask[h - crop_margin_h:, w - crop_margin_w:] = False
            grayA = grayA * mask  # Apply mask to grayscale image

            # Enhance contrast using adaptive histogram equalization
            grayA = exposure.equalize_adapthist(grayA)

            # Denoise the image using bilateral filtering
            grayA = cv2.bilateralFilter((grayA * 255).astype(np.uint8), 9, 75, 75)

            # Segment the image using Otsu's thresholding
            threshold = threshold_otsu(grayA)
            binary_A = (grayA < threshold).astype(np.uint8) * 255

            # Apply morphological operations to clean segmentation
            binary_A = morphology.remove_small_objects(binary_A, min_size=1600)#1000
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=10000)#4000
            binary_A = morphology.opening(binary_A)
            
            binary_A = (binary_A > 0).astype(np.uint8) * 255
            
            # Label connected regions
            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            # Create mask for excluding cropped scale bar area
            crop_start_row = h - crop_margin_h
            crop_start_col = w - crop_margin_w
            crop_mask = np.zeros_like(region_labels_A, dtype=bool)
            crop_mask[crop_start_row:, crop_start_col:] = True

            # Filter out regions that intersect with the cropped area
            filtered_labels = []
            for region in region_props_A:
                region_mask = (region_labels_A == region.label)
                if not np.any(region_mask & crop_mask):
                    filtered_labels.append(region.label)

            # Generate new label image without excluded regions
            new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
            label_counter = 1
            for lbl in filtered_labels:
                new_label_img[region_labels_A == lbl] = label_counter
                label_counter += 1

            # Refresh region labels and properties
            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            # Calculate region area statistics for filtering/splitting
            areas = [region.area for region in region_props_A]
            media_area = np.median(areas)
            std_area = np.std(areas)
            min_area = np.min(areas)
            average = min_area

            # --- Save histogram of region areas ---
            plt.figure(figsize=(8, 5))
            plt.hist(areas, bins=20, color='skyblue', edgecolor='black')
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
            print(f"Saved histogram for {bf_file} to {hist_areas_image_path}")
            all_output_files.append(hist_areas_image_path)

            # Refine label image: keep small regions, split large ones using watershed
            for region in region_props_A:
                if region.area < average:
                    new_label_img[region.slice][region.image] = label_counter
                    label_counter += 1
                else:
                    region_mask = np.zeros_like(region_labels_A, dtype=np.uint8)
                    region_mask[region.slice][region.image] = 1
                    distance = ndi.distance_transform_edt(region_mask)
                    coordinates = peak_local_max(distance, labels=region_mask, min_distance=5)
                    local_maxi = np.zeros_like(distance, dtype=bool)
                    local_maxi[tuple(coordinates.T)] = True
                    markers = label(local_maxi)
                    labels_ws = watershed(-distance, markers, mask=region_mask)
                    for ws_label in np.unique(labels_ws):
                        if ws_label == 0:
                            continue
                        mask = labels_ws == ws_label
                        new_label_img[mask] = label_counter
                        label_counter += 1

            # Final labeled image after splitting
            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            # 🔥 Reset labels to start from 1
            region_labels_A = label(region_labels_A > 0)
            region_props_A = regionprops(region_labels_A)

            # Ensure binary mask matches grayscale shape
            if binary_A.shape != grayA.shape:
                binary_A = resize(binary_A, grayA.shape, order=0, preserve_range=True, anti_aliasing=False)

            # --- Visualize segmentation ---
            plt.figure(figsize=(8, 8))
            plt.imshow(region_labels_A, cmap='nipy_spectral')
            plt.title('Segmentation')
            plt.axis('off')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()

            # Annotate region labels on binary image
            overlay_image = cv2.cvtColor((binary_A > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for region in regionprops(region_labels_A):
                y, x = region.centroid
                label_id = region.label
                cv2.putText(overlay_image, str(region.label), (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Save annotated segmentation image
            annotated_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented_Annotated.png")
            cv2.imwrite(annotated_path, overlay_image)
            print(f"Saved annotated image with labels to {annotated_path}")
            all_output_files.append(annotated_path)

            # Create binary mask with only valid detected regions
            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (
                        region_labels_A[min_row:max_row, min_col:max_col] == prop.label
                    )
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255

            # --- Save region statistics to Excel ---
            region_area = pd.DataFrame({
                "Region_Label": [region.label for region in region_props_A],
                "Region_Area (pixels)": [region.area for region in region_props_A],
                "Region_Area (µm²)": [region.area * (pixel_to_um ** 2) for region in region_props_A]
            })

            # Filter out regions with non-positive area (shouldn't happen, but safe check)
            region_area_df = region_area[region_area["Region_Area (pixels)"] > 800]

            total_area = region_area_df["Region_Area (µm²)"].sum()
            total_cells = region_area_df["Region_Label"].count()

            # Append summary rows
            region_area_df.loc["Total Area"] = ["", "Total Area", total_area]
            region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

            # Save region stats to Excel
            region_area_excel_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Region_Area_in_um2.xlsx")
            region_area_df.to_excel(region_area_excel_path, index=False)
            print(f"Saved region areas for {bf_file} to {region_area_excel_path}")

            # --- Plot histogram of pixel intensities ---
            plt.figure(figsize=(8, 6))
            plt.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()

            # Save the pixel intensity histogram
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()
            plt.close()
            print(f"Saved histogram for {bf_file} to {annotated_path}")
            all_output_files.append(hist_cells_image_path)

            del grayA, binary_A, region_labels_A, region_props_A, overlay_image, filtered_binary_A
            gc.collect()

        self.log("Processing complete!")
        
        # -----------------------------------------------------
        # Create a ZIP archive with all output histogram and annotated image files
        zip_path = os.path.join(self.output_folder, "All_Images_histograms.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in all_output_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))
                
        # Remove the original files after archiving
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


# -*- coding: utf-8 -*-
"""
ImageProcessingApp (v2)
- Adds tunable GUI controls for pre-processing, crystal detection, and cell segmentation
- Media presets (Ammonium chloride, Urea, Hypoxanthine, Custom)
- Preview mode (no files written)
- Correct µm/px math
- Boolean morphology
- Safer label visualization (label2rgb)
- Overlap without resizing
- Vectorized cell↔crystal mapping
- Auto-threshold on PL with FP/Mpix target + contrast verification

Drop-in replacement for your current script. Requires: PyQt5, numpy, opencv-python, scikit-image, scipy, pandas, matplotlib, xlsxwriter, openpyxl
"""

import os
import sys
import json
import zipfile
import gc
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd

# matplotlib backend: prefer Qt5Agg; fall back to Agg in headless
try:
    import matplotlib
    matplotlib.use("Qt5Agg")
except Exception:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QTextEdit, QInputDialog, QComboBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox
)

from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_li, threshold_otsu, threshold_isodata, threshold_sauvola
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import opening, remove_small_objects, remove_small_holes, disk
from skimage.util import img_as_float
from skimage import exposure
from skimage.segmentation import watershed
from skimage.feature import blob_log, peak_local_max
from scipy import ndimage as ndi

# ------------------------------ Utilities ------------------------------ #

def apply_brightness_contrast(img_float01, brightness=0, contrast=0):
    """Brightness/contrast on [0,1] float image. brightness/contrast in [-100,100]."""
    b = np.clip(brightness, -100, 100) / 100.0
    c = np.clip(contrast, -100, 100) / 100.0
    out = img_float01.copy()
    out = out + b  # shift
    out = 0.5 + (out - 0.5) * (1.0 + c * 2.0)  # scale around mid-gray
    return np.clip(out, 0.0, 1.0)

def clahe01(img_float01, clip_limit=2.0, tile_grid_size=(8,8)):
    cl = max(0.01, float(clip_limit))
    img_u8 = (np.clip(img_float01, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=tile_grid_size)
    out = clahe.apply(img_u8)
    return out.astype(np.float32) / 255.0

def bilateral01(img_float01, d=9, sigmaColor=75, sigmaSpace=75):
    img_u8 = (np.clip(img_float01, 0, 1) * 255).astype(np.uint8)
    out = cv2.bilateralFilter(img_u8, int(d), int(sigmaColor), int(sigmaSpace))
    return out.astype(np.float32) / 255.0

def label_vis_rgb(lbl):
    vis = label2rgb(lbl, bg_label=0)  # float [0..1]
    return (vis * 255).astype(np.uint8)[:, :, ::-1]  # BGR for cv2.imwrite

# ------------------------------ Main App ------------------------------ #

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.bf_folder = ""
        self.pl_folder = ""
        self.output_folder = ""
        self.processing_active = False

        # default scales (µm per pixel entries map name->µm/px value)
        self.um_to_px_map = {"40": 0.4, "100": 1.0}  # placeholders; user sets properly
        self.load_scale_settings()

        self.initUI()

    # ---------------- GUI ---------------- #
    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Scale block
        self.pixel_distance_label = QLabel("Distance in pixels:")
        self.pixel_distance_input = QLineEdit(); self.pixel_distance_input.setText("NOT VALUE")
        self.known_um_label = QLabel("Known distance (µm):")
        self.known_um_combo = QComboBox(); self.known_um_combo.setEditable(True)
        self.known_um_combo.addItems(sorted(self.um_to_px_map.keys()))
        self.known_um_combo.setCurrentText("NOT VALUE")
        self.known_um_combo.lineEdit().editingFinished.connect(self.on_custom_um_entered)

        self.set_scale_button = QPushButton("Set µm↔px from scalebar")
        self.set_scale_button.clicked.connect(self.set_known_um_and_px)
        self.delete_scale_button = QPushButton("Delete Selected Scale")
        self.delete_scale_button.clicked.connect(self.delete_selected_scale)

        for w in [self.set_scale_button, self.delete_scale_button,
                  self.pixel_distance_label, self.pixel_distance_input,
                  self.known_um_label, self.known_um_combo]:
            layout.addWidget(w)

        # folders
        self.bf_label = QLabel("BF Folder: Not selected")
        self.pl_label = QLabel("PL Folder: Not selected")
        self.output_label = QLabel("Output Folder: Not selected")
        self.bf_button = QPushButton("Select BF Folder"); self.bf_button.clicked.connect(self.select_bf_folder)
        self.pl_button = QPushButton("Select PL Folder"); self.pl_button.clicked.connect(self.select_pl_folder)
        self.output_button = QPushButton("Select Output Folder"); self.output_button.clicked.connect(self.select_output_folder)
        for w in [self.bf_label, self.bf_button, self.pl_label, self.pl_button, self.output_label, self.output_button]:
            layout.addWidget(w)

        # Tunable panels
        self.init_tunable_panels()
        layout.addWidget(self.pre_box)
        layout.addWidget(self.cr_box)
        layout.addWidget(self.cell_box)

        # Media profile
        prof_row = QHBoxLayout()
        self.profile_label = QLabel("Media profile:")
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["Ammonium chloride", "Urea", "Hypoxanthine", "Custom"])
        self.profile_combo.currentIndexChanged.connect(self.apply_profile_defaults)
        prof_row.addWidget(self.profile_label); prof_row.addWidget(self.profile_combo)
        layout.addLayout(prof_row)

        # Preset save/load
        pr_row = QHBoxLayout()
        self.save_preset_btn = QPushButton("Save preset…"); self.save_preset_btn.clicked.connect(self.save_preset)
        self.load_preset_btn = QPushButton("Load preset…"); self.load_preset_btn.clicked.connect(self.load_preset)
        pr_row.addWidget(self.save_preset_btn); pr_row.addWidget(self.load_preset_btn)
        layout.addLayout(pr_row)

        # Actions
        self.preview_btn = QPushButton("Preview (no save)"); self.preview_btn.clicked.connect(lambda: self.run_pipeline(preview=True))
        self.process_button = QPushButton("Crystals: % Cells with crystals"); self.process_button.clicked.connect(lambda: self.run_pipeline(mode=1))
        self.process_button_2 = QPushButton("Crystals: % Area crystals/cells"); self.process_button_2.clicked.connect(lambda: self.run_pipeline(mode=2))
        self.process_button_3 = QPushButton("Cells only: segmentation exports"); self.process_button_3.clicked.connect(lambda: self.run_pipeline(mode=3))
        layout.addWidget(self.preview_btn)
        layout.addWidget(self.process_button)
        layout.addWidget(self.process_button_2)
        layout.addWidget(self.process_button_3)

        # Log
        self.log_output = QTextEdit(); self.log_output.setReadOnly(True); layout.addWidget(self.log_output)

        self.setWindowTitle("Batch Image Processing (tunable)")
        self.resize(900, 800)

        # apply defaults for initial profile
        self.apply_profile_defaults()

    def init_tunable_panels(self):
        def dspin(lo, hi, step, val):
            w = QDoubleSpinBox(self); w.setRange(lo, hi); w.setSingleStep(step); w.setValue(val); return w
        def ispin(lo, hi, val):
            w = QSpinBox(self); w.setRange(lo, hi); w.setValue(val); return w

        # Pre-processing
        self.pre_box = QGroupBox("Pre-processing")
        f1 = QFormLayout()
        self.brightness = ispin(-100, 100, 0)
        self.contrast = ispin(-100, 100, 0)
        self.clahe = dspin(0.5, 6.0, 0.1, 2.0)
        self.bil_d = ispin(1, 15, 9)
        self.bil_sc = ispin(10, 200, 75)
        self.bil_ss = ispin(10, 200, 75)
        f1.addRow("Brightness", self.brightness)
        f1.addRow("Contrast", self.contrast)
        f1.addRow("CLAHE clip_limit", self.clahe)
        f1.addRow("Bilateral d", self.bil_d)
        f1.addRow("Bilateral sigmaColor", self.bil_sc)
        f1.addRow("Bilateral sigmaSpace", self.bil_ss)
        self.pre_box.setLayout(f1)

        # Crystal detection
        self.cr_box = QGroupBox("Crystal detection (PL)")
        f2 = QFormLayout()
        self.sigma_min_um = dspin(0.3, 12.0, 0.1, 0.6)
        self.sigma_max_um = dspin(0.5, 15.0, 0.1, 3.0)
        self.num_sigma = ispin(3, 12, 6)
        self.log_thresh = dspin(0.001, 0.1, 0.001, 0.02)
        self.min_area_um2 = dspin(0.1, 100.0, 0.1, 1.0)
        self.only_in_cells = QCheckBox(); self.only_in_cells.setChecked(True)
        self.auto_thresh = QCheckBox(); self.auto_thresh.setChecked(True)
        self.fp_target = dspin(0.0, 3.0, 0.05, 0.30)  # false positives per Mpix
        self.contrast_k = dspin(0.0, 3.0, 0.1, 1.2)
        self.ring_width_px = ispin(2, 20, 6)
        f2.addRow("sigma_min (µm)", self.sigma_min_um)
        f2.addRow("sigma_max (µm)", self.sigma_max_um)
        f2.addRow("num_sigma", self.num_sigma)
        f2.addRow("LoG threshold", self.log_thresh)
        f2.addRow("min_crystal_area (µm²)", self.min_area_um2)
        f2.addRow("Require overlap with cells", self.only_in_cells)
        f2.addRow("Auto-threshold", self.auto_thresh)
        f2.addRow("FP target (/Mpix)", self.fp_target)
        f2.addRow("Contrast k (blob vs ring)", self.contrast_k)
        f2.addRow("Ring width (px)", self.ring_width_px)
        self.cr_box.setLayout(f2)

        # Cell segmentation
        self.cell_box = QGroupBox("Cell segmentation (BF)")
        f3 = QFormLayout()
        self.th_method = QComboBox(); self.th_method.addItems(["Otsu", "Li", "Isodata", "Sauvola"])
        self.min_cell_area_um2 = dspin(1.0, 500.0, 1.0, 12.0)
        self.hole_area_um2 = dspin(5.0, 2000.0, 5.0, 60.0)
        self.open_size_px = ispin(0, 7, 1)
        self.min_dist_px = ispin(1, 40, 5)
        self.dt_sigma_px = dspin(0.0, 5.0, 0.5, 1.0)
        f3.addRow("Threshold method", self.th_method)
        f3.addRow("min_cell_area (µm²)", self.min_cell_area_um2)
        f3.addRow("remove_holes_area (µm²)", self.hole_area_um2)
        f3.addRow("Opening size (px)", self.open_size_px)
        f3.addRow("watershed min_distance (px)", self.min_dist_px)
        f3.addRow("DT blur sigma (px)", self.dt_sigma_px)
        self.cell_box.setLayout(f3)

    # ---------------- Scale management ---------------- #
    def on_custom_um_entered(self):
        text = self.known_um_combo.currentText().strip()
        if text and text not in [self.known_um_combo.itemText(i) for i in range(self.known_um_combo.count())]:
            self.known_um_combo.addItem(text)

    def set_known_um_and_px(self):
        known_um, ok1 = QInputDialog.getDouble(self, "Known µm", "Enter known micrometer value:", decimals=6)
        if not ok1:
            return
        distance_px, ok2 = QInputDialog.getDouble(self, "Distance in Pixels", "Enter distance in pixels:", decimals=6)
        if not ok2 or distance_px <= 0:
            return
        # store µm per pixel for this scale name (use the known µm as name)
        um_per_px = known_um / distance_px
        name = f"{known_um:g}"
        self.um_to_px_map[name] = um_per_px
        self.save_scale_settings(); self.load_scale_settings()
        self.known_um_combo.setCurrentText(name)
        QMessageBox.information(self, "Saved", f"Added mapping '{name}' = {um_per_px:.6f} µm/px")

    def delete_selected_scale(self):
        selected = self.known_um_combo.currentText().strip()
        if selected in self.um_to_px_map:
            del self.um_to_px_map[selected]
            self.save_scale_settings(); self.load_scale_settings()
            QMessageBox.information(self, "Deleted", f"Removed mapping '{selected}'.")
        else:
            QMessageBox.warning(self, "Not Found", "Scale not found.")

    def save_scale_settings(self):
        with open('scale_map.json', 'w') as f:
            json.dump(self.um_to_px_map, f, indent=2)

    def load_scale_settings(self):
        try:
            with open('scale_map.json', 'r') as f:
                self.um_to_px_map = json.load(f)
        except FileNotFoundError:
            pass

    # ---------------- Folders ---------------- #
    def select_bf_folder(self):
        self.bf_folder = QFileDialog.getExistingDirectory(self, "Select BF Folder")
        self.bf_label.setText(f"BF Folder: {self.bf_folder}")

    def select_pl_folder(self):
        self.pl_folder = QFileDialog.getExistingDirectory(self, "Select PL Folder")
        self.pl_label.setText(f"PL Folder: {self.pl_folder}")

    def select_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        self.output_label.setText(f"Output Folder: {self.output_folder}")

    # ---------------- Presets / Profiles ---------------- #
    def apply_profile_defaults(self):
        name = self.profile_combo.currentText()
        if name == "Ammonium chloride":
            self.sigma_min_um.setValue(0.6); self.sigma_max_um.setValue(3.0)
            self.min_area_um2.setValue(0.8)
        elif name == "Urea":
            self.sigma_min_um.setValue(0.6); self.sigma_max_um.setValue(3.5)
            self.min_area_um2.setValue(1.0)
        elif name == "Hypoxanthine":
            self.sigma_min_um.setValue(1.0); self.sigma_max_um.setValue(5.0)
            self.min_area_um2.setValue(2.0)
        # else Custom: leave as-is

    def save_preset(self):
        preset = self._collect_params()
        with open('tunable_preset.json', 'w') as f:
            json.dump(preset, f, indent=2)
        QMessageBox.information(self, "Preset", "Saved to tunable_preset.json")

    def load_preset(self):
        try:
            with open('tunable_preset.json', 'r') as f:
                p = json.load(f)
            self._apply_params(p)
            QMessageBox.information(self, "Preset", "Loaded from tunable_preset.json")
        except Exception as e:
            QMessageBox.warning(self, "Preset", f"Failed to load: {e}")

    def _collect_params(self):
        return {
            "brightness": self.brightness.value(),
            "contrast": self.contrast.value(),
            "clahe": self.clahe.value(),
            "bil_d": self.bil_d.value(),
            "bil_sc": self.bil_sc.value(),
            "bil_ss": self.bil_ss.value(),
            "sigma_min_um": self.sigma_min_um.value(),
            "sigma_max_um": self.sigma_max_um.value(),
            "num_sigma": self.num_sigma.value(),
            "log_thresh": self.log_thresh.value(),
            "min_area_um2": self.min_area_um2.value(),
            "only_in_cells": self.only_in_cells.isChecked(),
            "auto_thresh": self.auto_thresh.isChecked(),
            "fp_target": self.fp_target.value(),
            "contrast_k": self.contrast_k.value(),
            "ring_width_px": self.ring_width_px.value(),
            "th_method": self.th_method.currentText(),
            "min_cell_area_um2": self.min_cell_area_um2.value(),
            "hole_area_um2": self.hole_area_um2.value(),
            "open_size_px": self.open_size_px.value(),
            "min_dist_px": self.min_dist_px.value(),
            "dt_sigma_px": self.dt_sigma_px.value(),
        }

    def _apply_params(self, p):
        self.brightness.setValue(int(p.get("brightness",0)))
        self.contrast.setValue(int(p.get("contrast",0)))
        self.clahe.setValue(float(p.get("clahe",2.0)))
        self.bil_d.setValue(int(p.get("bil_d",9)))
        self.bil_sc.setValue(int(p.get("bil_sc",75)))
        self.bil_ss.setValue(int(p.get("bil_ss",75)))
        self.sigma_min_um.setValue(float(p.get("sigma_min_um",0.6)))
        self.sigma_max_um.setValue(float(p.get("sigma_max_um",3.0)))
        self.num_sigma.setValue(int(p.get("num_sigma",6)))
        self.log_thresh.setValue(float(p.get("log_thresh",0.02)))
        self.min_area_um2.setValue(float(p.get("min_area_um2",1.0)))
        self.only_in_cells.setChecked(bool(p.get("only_in_cells", True)))
        self.auto_thresh.setChecked(bool(p.get("auto_thresh", True)))
        self.fp_target.setValue(float(p.get("fp_target", 0.3)))
        self.contrast_k.setValue(float(p.get("contrast_k", 1.2)))
        self.ring_width_px.setValue(int(p.get("ring_width_px", 6)))
        t = p.get("th_method", "Otsu")
        idx = max(0, self.th_method.findText(t))
        self.th_method.setCurrentIndex(idx)
        self.min_cell_area_um2.setValue(float(p.get("min_cell_area_um2", 12.0)))
        self.hole_area_um2.setValue(float(p.get("hole_area_um2", 60.0)))
        self.open_size_px.setValue(int(p.get("open_size_px", 1)))
        self.min_dist_px.setValue(int(p.get("min_dist_px", 5)))
        self.dt_sigma_px.setValue(float(p.get("dt_sigma_px", 1.0)))

    # ---------------- Logging ---------------- #
    def log(self, msg):
        self.log_output.append(str(msg))

    # ---------------- Pipeline ---------------- #
    def _um_per_px_from_gui(self):
        try:
            distance_px = float(self.pixel_distance_input.text())
            known_um = float(self.known_um_combo.currentText())
            if distance_px <= 0: raise ValueError
            return known_um / distance_px
        except Exception:
            raise ValueError("Please set valid scalebar values (known µm and distance in pixels).")

    def run_pipeline(self, mode=1, preview=False):
        # mode 1: % cells with crystals; mode 2: % area crystals/cells; mode 3: cells only
        try:
            um_per_px = self._um_per_px_from_gui()
        except ValueError as e:
            QMessageBox.warning(self, "Scale", str(e)); return

        if not self.bf_folder or not self.pl_folder:
            self.log("Select BF and PL folders first."); return
        if not preview and not self.output_folder:
            self.log("Select Output folder for saving results."); return

        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.lower().endswith('.tif')])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.lower().endswith('.tif')])
        if len(bf_files) != len(pl_files):
            QMessageBox.warning(self, "Files", "Mismatch between BF and PL counts."); return

        if not preview:
            os.makedirs(self.output_folder, exist_ok=True)

        summary_rows = []
        all_output_files = []

        params = self._collect_params()
        self.log(f"Params: {params}")

        for bf_file, pl_file in zip(bf_files, pl_files):
            stem = os.path.splitext(bf_file)[0]
            self.log(f"Processing {bf_file} + {pl_file}")

            imageA = cv2.imread(os.path.join(self.bf_folder, bf_file), cv2.IMREAD_COLOR)
            imageB = cv2.imread(os.path.join(self.pl_folder, pl_file), cv2.IMREAD_COLOR)
            if imageA is None or imageB is None:
                self.log("Could not read one of the images, skipping."); continue

            # --- BF to grayscale float [0,1]
            grayA = rgb2gray(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)).astype(np.float32)
            # mask out scale bar (bottom-right 1.5% x 2.5%)
            h, w = grayA.shape
            crop_h = int(0.015 * h); crop_w = int(0.025 * w)
            grayA_mask = np.ones_like(grayA, dtype=bool)
            grayA_mask[h-crop_h:, w-crop_w:] = False
            grayA = np.where(grayA_mask, grayA, 0)
            # enhance
            grayA = clahe01(grayA, clip_limit=self.clahe.value())
            grayA = bilateral01(grayA, d=self.bil_d.value(), sigmaColor=self.bil_sc.value(), sigmaSpace=self.bil_ss.value())

            # --- Cell thresholding
            method = self.th_method.currentText()
            if method == "Otsu":
                th = threshold_otsu((grayA*255).astype(np.uint8)) / 255.0
                cells = (grayA < th)
            elif method == "Li":
                th = threshold_li(grayA)
                cells = (grayA < th)
            elif method == "Isodata":
                th = threshold_isodata(grayA)
                cells = (grayA < th)
            else:  # Sauvola
                win = max(15, int(round(min(h,w)*0.03))//2*2+1)
                sau = threshold_sauvola(grayA, window_size=win)
                cells = (grayA < sau)

            # Morphology on boolean
            min_cell_area_px = int(round(self.min_cell_area_um2.value() / (um_per_px**2)))
            hole_area_px = int(round(self.hole_area_um2.value() / (um_per_px**2)))
            if min_cell_area_px > 0:
                cells = remove_small_objects(cells, min_size=min_cell_area_px)
            if hole_area_px > 0:
                cells = remove_small_holes(cells, area_threshold=hole_area_px)
            k = self.open_size_px.value()
            if k > 0:
                cells = opening(cells, selem=disk(k))

            # Split (watershed)
            dt = ndi.distance_transform_edt(cells)
            if self.dt_sigma_px.value() > 0:
                dt = cv2.GaussianBlur(dt, (0,0), self.dt_sigma_px.value())
            coords = peak_local_max(dt, labels=cells.astype(np.uint8), min_distance=self.min_dist_px.value())
            seeds = np.zeros_like(dt, dtype=np.int32)
            seeds[tuple(coords.T)] = np.arange(1, coords.shape[0]+1)
            labels_cells = watershed(-dt, seeds, mask=cells)

            # --- PL processing
            grayB = rgb2gray(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)).astype(np.float32)
            grayB = exposure.rescale_intensity(grayB, in_range='image', out_range=(0,1)).astype(np.float32)
            grayB = clahe01(grayB, clip_limit=self.clahe.value())
            grayB = apply_brightness_contrast(grayB, self.brightness.value(), self.contrast.value())
            grayB = bilateral01(grayB, d=self.bil_d.value(), sigmaColor=self.bil_sc.value(), sigmaSpace=self.bil_ss.value())

            # Candidate LoG
            sigma_min_px = max(0.5, self.sigma_min_um.value() / um_per_px)
            sigma_max_px = max(sigma_min_px+0.1, self.sigma_max_um.value() / um_per_px)
            threshold_val = self.log_thresh.value()

            def detect_with_threshold(t):
                return blob_log(grayB, min_sigma=sigma_min_px, max_sigma=sigma_max_px,
                                num_sigma=self.num_sigma.value(), threshold=t)

            blobs = detect_with_threshold(threshold_val)

            # Auto-threshold: limit FP density on background (outside cells)
            if self.auto_thresh.isChecked():
                bg_mask = ~cells
                # count candidates whose centers fall in background
                def fp_density(blobs_arr):
                    if blobs_arr.size == 0: return 0.0
                    yy = blobs_arr[:,0].astype(int); xx = blobs_arr[:,1].astype(int)
                    yy = np.clip(yy, 0, h-1); xx = np.clip(xx, 0, w-1)
                    fp = np.count_nonzero(bg_mask[yy, xx])
                    mpix = (h*w) / 1_000_000.0
                    return fp / max(mpix, 1e-6)
                cur = fp_density(blobs)
                while cur > self.fp_target.value() and threshold_val < 0.1:
                    threshold_val *= 1.10
                    blobs = detect_with_threshold(threshold_val)
                    cur = fp_density(blobs)

            # Verify candidates: contrast test + area + overlap option
            crystals_mask = np.zeros_like(grayB, dtype=bool)
            ring_w = int(self.ring_width_px.value())
            k_con = float(self.contrast_k.value())
            min_area_px = int(round(self.min_area_um2.value() / (um_per_px**2)))

            for (y, x, sigma) in blobs:
                r = int(max(1, np.sqrt(2.0) * float(sigma)))
                yy, xx = np.ogrid[:h, :w]
                circle = (yy - int(y))**2 + (xx - int(x))**2 <= r*r
                ring = (yy - int(y))**2 + (xx - int(x))**2 <= (r+ring_w)**2
                ann = np.logical_and(ring, ~circle)
                if not np.any(circle) or not np.any(ann):
                    continue
                m_in = float(np.mean(grayB[circle])); m_out = float(np.mean(grayB[ann]))
                sd_out = float(np.std(grayB[ann])) + 1e-6
                if m_in >= (m_out + k_con * sd_out):
                    # pass contrast test
                    if self.only_in_cells.isChecked():
                        # center must be inside cell
                        cy = int(np.clip(int(y), 0, h-1)); cx = int(np.clip(int(x), 0, w-1))
                        if not cells[cy, cx]:
                            continue
                    # area filter (approx circle)
                    area_px = np.count_nonzero(circle)
                    if area_px >= max(1, min_area_px):
                        crystals_mask[circle] = True

            # Final masks/labels
            overlap_mask = crystals_mask & cells
            labels_cells = label(labels_cells > 0)
            labels_crystals = label(overlap_mask)

            # Vectorized mapping crystal→cell via bincount on cell labels under each crystal region
            mapping = []
            total_area_cells_um2 = float(np.count_nonzero(cells)) * (um_per_px**2)
            total_area_cr_um2 = float(np.count_nonzero(overlap_mask)) * (um_per_px**2)

            for lab in range(1, labels_crystals.max()+1):
                reg = (labels_crystals == lab)
                under = labels_cells[reg]
                under = under[under > 0]
                if under.size == 0:
                    best_cell, overlap_px = None, 0
                else:
                    counts = np.bincount(under)
                    best_cell = int(np.argmax(counts))
                    overlap_px = int(counts[best_cell])
                mapping.append({
                    "Region_Label": int(lab),
                    "Associated_Cell": best_cell,
                    "Overlap (pixels)": overlap_px,
                    "Region_Area (pixels)": int(np.count_nonzero(reg)),
                    "Region_Area (µm²)": float(np.count_nonzero(reg)) * (um_per_px**2)
                })

            df_map = pd.DataFrame(mapping)
            df_map = df_map[(df_map["Region_Area (µm²)"] > 0) & (df_map["Overlap (pixels)"] > 0)]
            total_distinct_cells = df_map["Associated_Cell"].nunique() if not df_map.empty else 0

            # --- Exports / Summary
            # cell areas table
            props_cells = regionprops_table(labels_cells, properties=("label", "area"))
            df_cells = pd.DataFrame(props_cells)
            df_cells["Area (µm²)"] = df_cells["area"] * (um_per_px**2)
            df_cells = df_cells[df_cells["area"] > max(1, min_cell_area_px)]

            if mode == 1:
                percentage = (total_distinct_cells / max(1, df_cells.shape[0])) * 100.0
                summary_rows.append({"Days": stem, "total_cells": int(df_cells.shape[0]),
                                     "cells_with_crystals": int(total_distinct_cells),
                                     "%_cells_with_crystals": percentage})
            elif mode == 2:
                pct = (total_area_cr_um2 / max(1e-9, total_area_cells_um2)) * 100.0
                summary_rows.append({"Days": stem, "total_cells_area": total_area_cells_um2,
                                     "total_crystals_area": total_area_cr_um2,
                                     "%_area_crystals_cells": pct})

            if not preview:
                # Visualizations
                seg_bgr = label_vis_rgb(labels_cells)
                cry_bgr = label_vis_rgb(labels_crystals)
                ann = imageA.copy()
                ann[overlap_mask] = (0,255,255)  # highlight overlap in yellow

                cv2.imwrite(os.path.join(self.output_folder, f"{stem}_cells_seg.png"), seg_bgr)
                cv2.imwrite(os.path.join(self.output_folder, f"{stem}_crystals_overlap.png"), cry_bgr)
                cv2.imwrite(os.path.join(self.output_folder, f"{stem}_annotated.png"), ann)

                # Excel with 3 sheets
                xlsx_path = os.path.join(self.output_folder, f"{stem}_All_Datasets.xlsx")
                with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
                    df_cells.rename(columns={"label":"Region_Label","area":"Region_Area (pixels)"}).to_excel(writer, sheet_name='Cells', index=False)
                    df_map.to_excel(writer, sheet_name='Crystals', index=False)
                    # Crystal sizes
                    props_cr = regionprops_table(labels_crystals, properties=("label","area"))
                    df_cr = pd.DataFrame(props_cr)
                    df_cr["area_um2"] = df_cr["area"] * (um_per_px**2)
                    df_cr["eq_diam_um"] = 2.0 * np.sqrt(df_cr["area_um2"]/np.pi)
                    df_cr.to_excel(writer, sheet_name='Crystal sizes', index=False)

        # After loop: plots + summary exports
        if preview:
            self.log("Preview complete (no files written).")
            return

        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            df_sum["Days"] = df_sum["Days"].astype(str)
            df_sum = df_sum.sort_values("Days")

            # Plot based on mode
            if mode == 1:
                ykey = "%_cells_with_crystals"
            elif mode == 2:
                ykey = "%_area_crystals_cells"
            else:
                ykey = None

            if ykey and ykey in df_sum.columns:
                # Convert to numeric if needed
                df_sum[ykey] = pd.to_numeric(df_sum[ykey])
                # Extract numeric day
                df_sum["Days_num"] = pd.to_numeric(df_sum["Days"].str.extract(r"(\d+)")[0], errors='coerce')
                g = df_sum.groupby("Days_num")[ykey].agg(['mean','std']).reset_index().dropna()

                plt.figure(figsize=(9,5))
                plt.plot(g["Days_num"], g['mean'], marker='o', linewidth=2, label='Average')
                for x, m, s in zip(g["Days_num"], g['mean'], g['std'].fillna(0)):
                    plt.vlines(x, m-s, m+s, linewidth=2, label='±1 STD' if x == g["Days_num"].iloc[0] else "")
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles)); plt.legend(by_label.values(), by_label.keys())
                plt.xlabel("Days"); plt.ylabel(ykey); plt.title(ykey)
                plt.grid(True); plt.tight_layout()
                plot_path = os.path.join(self.output_folder, "Plot.png")
                plt.savefig(plot_path, dpi=300); plt.close()
                g.rename(columns={'mean':'mean_percentage','std':'std_percentage'}).to_excel(os.path.join(self.output_folder, "Plot.xlsx"), index=False)

        self.log("Processing complete.")

    # ---------------- End Pipeline ---------------- #


# ------------------------------ Entry ------------------------------ #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ImageProcessingApp()
    w.show()
    sys.exit(app.exec_())

