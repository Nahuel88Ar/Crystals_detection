#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import sys
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.use("Qt5Agg")
from PyQt5.QtWidgets import QComboBox

from skimage.measure import label, regionprops
from skimage.filters import threshold_li
from skimage.filters import threshold_otsu
from skimage.filters import threshold_isodata
from skimage import data, filters, measure, morphology
from skimage.color import rgb2gray
from skimage.morphology import opening, remove_small_objects, remove_small_holes, disk
from skimage import morphology, exposure
from skimage import color
from skimage.feature import peak_local_max
from skimage.segmentation import morphological_chan_vese
from skimage.segmentation import slic
from skimage.segmentation import active_contour
from skimage.segmentation import watershed
from skimage.io import imread
from skimage.transform import resize
from skimage import draw
from skimage.segmentation import random_walker

from sklearn.cluster import KMeans

import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt, label as ndi_label
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from scipy.signal import find_peaks

from xlsxwriter import Workbook

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,QTextEdit,QInputDialog

QApplication.processEvents()

from threading import Event

from collections import defaultdict

import zipfile

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.um_to_px_map = {
            "40": 5.64039652,
            "100": 13.889
        }
        self.bf_folder = ""
        self.pl_folder = ""
        self.output_folder = ""
        self.processing_active = False  # Track if processing is active
        self.stop_event = Event()  # Event to handle stopping
        #self.um_to_px_map = self.load_um_to_px_mapping()
        self.load_scale_settings()

    def initUI(self):
        # GUI Layout and Buttons
        layout = QVBoxLayout()

        # Input for distance in pixels
        self.pixel_distance_label = QLabel("Distance in pixels:")
        self.pixel_distance_input = QLineEdit()
        self.pixel_distance_input.setText("NOT VALUE")

        # Input for known distance in micrometers
        self.known_um_label = QLabel("Known distance (µm):")
        self.known_um_combo = QComboBox()
        self.known_um_combo.setEditable(True)
        self.known_um_combo.addItems(["40", "100"])
        self.known_um_combo.setCurrentText("NOT VALUE")
        self.known_um_combo.setInsertPolicy(QComboBox.InsertAtBottom)
        self.known_um_combo.lineEdit().editingFinished.connect(self.on_custom_um_entered)
        self.known_um_combo.currentIndexChanged.connect(self.update_pixel_distance)
        
        self.bf_label = QLabel("BF Folder: Not selected")
        self.pl_label = QLabel("PL Folder: Not selected")
        self.output_label = QLabel("Output Folder: Not selected")

        self.set_scale_button = QPushButton("Set µm to px Scale")
        self.bf_button = QPushButton("Select BF Folder")
        self.pl_button = QPushButton("Select PL Folder")
        self.output_button = QPushButton("Select Output Folder")
        self.process_button_3 = QPushButton("Number of cells")
        self.stop_button = QPushButton("Stop Processing")
        self.restart_button = QPushButton("Restart Processing")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        
        self.set_scale_button.clicked.connect(self.set_known_um_and_px)
        self.bf_button.clicked.connect(self.select_bf_folder)
        self.pl_button.clicked.connect(self.select_pl_folder)
        self.output_button.clicked.connect(self.select_output_folder)
        self.process_button_3.clicked.connect(self.start_processing_3)
        self.stop_button.clicked.connect(self.stop_processing)
        self.restart_button.clicked.connect(self.restart_processing)

        layout.addWidget(self.set_scale_button)  # or add to wherever your layout is
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
        layout.addWidget(self.process_button_3)
        layout.addWidget(self.log_output)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.restart_button)
        
        self.setLayout(layout)
        self.setWindowTitle("Batch Image Processing")
        self.resize(500, 400)
    
    def log(self, message):
        self.log_output.append(message)

    def on_custom_um_entered(self):
        text = self.known_um_combo.currentText().strip()
        #if not text.endswith("um"):
        #    text += " um"
        if text not in [self.known_um_combo.itemText(i) for i in range(self.known_um_combo.count())]:
            self.known_um_combo.addItem(text)

    def update_pixel_distance(self):
        text = self.known_um_combo.currentText()
        if text in self.um_to_px_map:
            self.pixel_distance_input.setText(str(self.um_to_px_map[text]))
        else:
            self.pixel_distance_input.clear()
    
    def select_bf_folder(self):
        self.bf_folder = QFileDialog.getExistingDirectory(self, "Select BF Folder")
        self.bf_label.setText(f"BF Folder: {self.bf_folder}")
    
    def select_pl_folder(self):
        self.pl_folder = QFileDialog.getExistingDirectory(self, "Select PL Folder")
        self.pl_label.setText(f"PL Folder: {self.pl_folder}")
    
    def select_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        self.output_label.setText(f"Output Folder: {self.output_folder}")

    def stop_processing(self):
        self.stop_event.set()  # Set stop flag
        self.log("Stopping process...")

    def restart_processing(self):
        self.stop_processing()  # Stop current process
        self.log("Restarting processing...")
        self.start_processing_3()  # Start again
    
    def set_known_um_and_px(self):
        known_um, ok1 = QInputDialog.getDouble(self, "Known µm", "Enter known micrometer value:", decimals=6)
        if not ok1:
            return

        distance_px, ok2 = QInputDialog.getDouble(self, "Distance in Pixels", "Enter distance in pixels:", decimals=6)
        if not ok2 or distance_px == 0:
            return

        um_per_px = known_um / distance_px
        name, ok3 = QInputDialog.getText(self, "Mapping Name", "Enter a name for this scale mapping:")
        if not ok3 or not name.strip():
            name = f"{known_um}um_{distance_px}px"
            #name = f"{known_um} um"

        self.um_to_px_map[name] = um_per_px
        self.save_scale_settings()

        QMessageBox.information(self, "Saved", f"Added mapping '{name}' = {um_per_px:.6f} µm/px")
    

    def save_scale_settings(self):
        #with open("scale_config.json", "w") as f:
        #    json.dump(self.um_to_px_map, f, indent=4)
        with open('scale_map.json', 'w') as f:
            json.dump(self.um_to_px_map, f)


    def load_scale_settings(self):
        #if os.path.exists("scale_config.json"):
        #    with open("scale_config.json", "r") as f:
        #        self.um_to_px_map = json.load(f)
        #else:
        #    self.um_to_px_map = {}
        try:
            with open('scale_map.json', 'r') as f:
                self.um_to_px_map = json.load(f)
        except FileNotFoundError:
            # fallback to defaults
            self.um_to_px_map = {
                "40": 5.64039652,
                "100": 13.889
            }

            self.known_um_combo.clear()
            self.known_um_combo.addItems(self.um_to_px_map.keys())
            #self.known_um_combo.setCurrentText("NOT VALUE")

    def add_new_scale(self, scale_name, scale_value):
        self.um_to_px_map[scale_name] = scale_value
        self.save_scale_settings()
    
    def start_processing_3(self):
        self.processing_active = True
        self.stop_event.clear()  # Reset stop event
        
        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        
        self.log("Starting batch processing...")
        
        try:
            distance_in_px = float(self.pixel_distance_input.text())
            known_um = float(self.known_um_combo.currentText())
            if distance_in_px == 0:
                raise ValueError("Distance in pixels cannot be zero.")
            pixel_to_um = 1 / (known_um / distance_in_px)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for distance in pixels and known µm.")
            return 
        
      
        # Create output folder if it does not exist
        os.makedirs(self.output_folder, exist_ok=True)

        # List all .tif files in the folders
        bf_files = sorted([f for f in os.listdir(self.bf_folder) if f.endswith('.tif')])
        pl_files = sorted([f for f in os.listdir(self.pl_folder) if f.endswith('.tif')])

        # Ensure that the number of BF and PL images match
        if len(bf_files) != len(pl_files):
            raise ValueError("Mismatch in the number of BF and PL .tif files.")
        
        all_output_files = []
        
        # Batch process each pair of BF and PL images
        for bf_file, pl_file in zip(bf_files, pl_files):
            print(f"Processing: {bf_file} and {pl_file}")

            if self.stop_event.is_set():  # Check if stop was requested
                self.log("Processing stopped.")
                return
            
            self.log(f"Processing {bf_file} and {pl_file}...")
    
            # Load the images
            bf_image_path = os.path.join(self.bf_folder, bf_file)
            pl_image_path = os.path.join(self.pl_folder, pl_file)
            imageA = cv2.imread(bf_image_path)
            imageB = cv2.imread(pl_image_path)

            # Ensure images are loaded correctly
            if imageA is None or imageB is None:
                print(f"Skipping {bf_file} or {pl_file}: Unable to load image.")
                continue
        
            # Convert BF image to grayscale and enhance contrast
            grayA = rgb2gray(imageA)
            # Adaptive histogram equalization
            grayA = exposure.equalize_adapthist(grayA)
            # Noise reduction
            grayA = cv2.bilateralFilter((grayA * 255).astype(np.uint8), 9, 75, 75)
            # Compute threshold using Li's method
            threshold = threshold_otsu(grayA)
            # Apply thresholding
            binary_A = (grayA < threshold).astype(np.uint8) * 255
    
            # Apply morphological operations to clean up the binary mask
            binary_A = morphology.opening(binary_A)
            binary_A = morphology.remove_small_objects(binary_A.astype(bool), min_size=500)
            binary_A = morphology.dilation(binary_A, morphology.disk(4))
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=5000)
            binary_A = morphology.closing(binary_A, morphology.disk(4))
            binary_A = (binary_A > 0).astype(np.uint8) * 255
    
            #Label connected regions in binary mask
            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
            label_counter = 1

            # Compute average threshold based on the mean and standart desviation of region area
            #max_area = max([region.area for region in region_props_A])
            #areas = [region.area for region in region_props_A]
            #areas_filtered = [region.area for region in region_props_A if region.area < 17500]

            areas = np.array([region.area for region in region_props_A])

            # Calculate the 90th percentile (cutoff to exclude top 5%)
            cutoff = np.percentile(areas, 95)

            # Filter areas below this cutoff
            areas_filtered = areas[areas < cutoff]
            
            mean_area = np.mean(areas_filtered)
            std_area = np.std(areas_filtered)
            average = mean_area + std_area

            # Plot histogram
            plt.figure(figsize=(8, 5))
            plt.hist(areas, bins=20, color='skyblue', edgecolor='black')
            plt.title("Histogram of Region Areas")
            plt.xlabel("Area (pixels)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            #plt.show()

            # Save the histogram image
            hist_areas_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histograms_areas.png")
            plt.savefig(hist_areas_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            print(f"Saved histogram for {bf_file} to {hist_areas_image_path}")
            all_output_files.append(hist_areas_image_path)

            #<1500,>=3,=2,=5,=42
            for region in region_props_A:
                if region.area < average:
                    new_label_img[region.slice][region.image] = label_counter
                    label_counter += 1
                else:
                    coords = np.column_stack(np.where(region.image))
                    if len(coords) >= 10:
                        n_clusters = 5
                        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(coords)
                        for k in range(n_clusters):
                            mask = (kmeans.labels_ == k)
                            subcoords = coords[mask]
                            for (r, c) in subcoords:
                                new_label_img[region.slice][r, c] = label_counter
                            label_counter += 1

            region_labels_A = new_label_img
            region_props_A = regionprops(region_labels_A)

            # Ensure binary_A is the correct shape (resize if necessary)
            if binary_A.shape != grayA.shape:
                binary_A = resize(binary_A, grayA.shape, order=0, preserve_range=True, anti_aliasing=False)
            
            # Visualize results
            plt.figure(figsize=(8,8))
            plt.imshow(region_labels_A, cmap='nipy_spectral')
            plt.title('Segmentation')
            plt.axis('off')
            #plt.show()
            plt.pause(0.001)
            QApplication.processEvents()

            # Convert label image to RGB for annotation
            overlay_image = cv2.cvtColor((binary_A > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

            # Loop through each region and annotate label number
            for region in regionprops(region_labels_A):
                y, x = region.centroid  # Note: (row, col) = (y, x)
                label_id = region.label
                cv2.putText(
                    overlay_image,
                    str(label_id),
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),  # Red color for text
                    1,
                    cv2.LINE_AA
                )
            
            # Save the annotated image
            annotated_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented_Annotated.png")
            cv2.imwrite(annotated_path, overlay_image)
            print(f"Saved annotated image with labels to {annotated_path}")
            all_output_files.append(annotated_path)
            
            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (
                        region_labels_A[min_row:max_row, min_col:max_col] == prop.label
                    )

            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255
            # Create a DataFrame for the regions with their area in µm²
            region_area = pd.DataFrame({
                "Region_Label": [region.label for region in region_props_A],
                "Region_Area (pixels)": [region.area for region in region_props_A],
                "Region_Area (µm²)": [region.area * (pixel_to_um ** 2) for region in region_props_A]
            })
    
            # Filter the DataFrame to keep only regions with an area smaller than 200 µm²  
            region_area_df = region_area[region_area["Region_Area (µm²)"] > 0]

            # Calculate the total area in µm²
            total_area = region_area_df["Region_Area (µm²)"].sum()

            # Add the total area to the DataFrame
            total_cells = region_area_df["Region_Label"].count() - 1  # Subtract 1 if you're excluding the bottom-right region
            region_area_df.loc["Total Area"] = ["", "Total Area", total_area]
            region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

            # Save the DataFrame to a CSV file
            region_area_excel_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Region_Area_in_um2.xlsx")
            region_area_df.to_excel(region_area_excel_path, index=False)

            print(f"Saved region areas for {bf_file} to {region_area_excel_path}")
    
            plt.figure(figsize=(8, 6))
            plt.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (A) = {threshold:.2f}')
            #plt.axvline(auto_percentile, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (A) = {auto_percentile:.2f}')
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
    
            # Save the histogram image
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            print(f"Saved histogram for {bf_file} to {annotated_path}")
            all_output_files.append(hist_cells_image_path)

        self.log("Processing complete!")
        
        # Zip all collected output files
        zip_path = os.path.join(self.output_folder, "All_Images_histograms.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in all_output_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))
                
        # Optionally delete the individual files after zipping
        for file_path in all_output_files:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())


# In[ ]:




