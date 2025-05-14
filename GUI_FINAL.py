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
matplotlib.use("Qt5Agg")

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

#from sklearn.cluster import KMeans

from scipy.ndimage import distance_transform_edt, label as ndi_label
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from scipy.signal import find_peaks

from xlsxwriter import Workbook

#from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,QTextEdit

from PyQt5.QtWidgets import QApplication
QApplication.processEvents()

from threading import Event

from collections import defaultdict

import zipfile

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.bf_folder = ""
        self.pl_folder = ""
        self.output_folder = ""
        self.processing_active = False  # Track if processing is active
        self.stop_event = Event()  # Event to handle stopping
    
    def initUI(self):
        # GUI Layout and Buttons
        layout = QVBoxLayout()

        # Input for distance in pixels
        self.pixel_distance_label = QLabel("Distance in pixels:")
        self.pixel_distance_input = QLineEdit()
        self.pixel_distance_input.setText("1.0")

        # Input for known distance in micrometers
        self.known_um_label = QLabel("Known distance (µm):")
        self.known_um_input = QLineEdit()
        self.known_um_input.setText("1.0")
        
        self.bf_label = QLabel("BF Folder: Not selected")
        self.pl_label = QLabel("PL Folder: Not selected")
        self.output_label = QLabel("Output Folder: Not selected")
        
        self.bf_button = QPushButton("Select BF Folder")
        self.pl_button = QPushButton("Select PL Folder")
        self.output_button = QPushButton("Select Output Folder")
        self.process_button = QPushButton("Start Processing Script 1")
        self.process_button_2 = QPushButton("Start Processing Script 2")
        self.stop_button = QPushButton("Stop Processing")
        self.restart_button = QPushButton("Restart Processing")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        
        self.bf_button.clicked.connect(self.select_bf_folder)
        self.pl_button.clicked.connect(self.select_pl_folder)
        self.output_button.clicked.connect(self.select_output_folder)
        self.process_button.clicked.connect(self.start_processing)
        self.process_button_2.clicked.connect(self.start_processing_2)
        self.stop_button.clicked.connect(self.stop_processing)
        self.restart_button.clicked.connect(self.restart_processing)

        layout.addWidget(self.pixel_distance_label)
        layout.addWidget(self.pixel_distance_input)
        layout.addWidget(self.known_um_label)
        layout.addWidget(self.known_um_input)
        layout.addWidget(self.bf_label)
        layout.addWidget(self.bf_button)
        layout.addWidget(self.pl_label)
        layout.addWidget(self.pl_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.process_button_2)
        layout.addWidget(self.log_output)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.restart_button)
        
        self.setLayout(layout)
        self.setWindowTitle("Batch Image Processing")
        self.resize(500, 400)
    
    def log(self, message):
        self.log_output.append(message)
    
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
        self.start_processing()  # Start again
    
    def start_processing(self):
        self.processing_active = True
        self.stop_event.clear()  # Reset stop event
        
        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        
        self.log("Starting batch processing...")
        # Define image scale: 7.0917 pixels = 1 µm
        #pixel_to_um = 1 / 7.0917
        try:
            distance_in_px = float(self.pixel_distance_input.text())
            known_um = float(self.known_um_input.text())
            if distance_in_px == 0:
                raise ValueError("Distance in pixels cannot be zero.")
            pixel_to_um = 1/(known_um / distance_in_px)
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

            # Ensure binary_A is the correct shape (resize if necessary)
            if binary_A.shape != grayA.shape:
                binary_A = resize(binary_A, grayA.shape, order=0, preserve_range=True, anti_aliasing=False)

    
            # Apply the Euclidean Distance Transform for watershed segmentation and helps identify the centers of objects(cells).
            distance = distance_transform_edt(binary_A)

            # Find local maxima in the distance image to use as markers, these peaks correspond to cell centers.
            local_maxi = peak_local_max(distance, labels=binary_A, min_distance=1)

            # Label the local maxima,in other words,labels each detected peak with a unique number.
            #These markers will be used as starting points for the watershed algorithm.
            # Initialize markers array (same shape as the image)
            markers = np.zeros_like(distance, dtype=int)
            #markers[local_maxi] = np.arange(1, np.sum(local_maxi) + 1)  # Label the markers

            #markers = np.zeros_like(distance, dtype=int)

            # Assign unique markers to the local maxima positions
            for i, (row, col) in enumerate(local_maxi):
                markers[row, col] = i + 1  # Start from 1

            # Apply watershed
            #Watershed treats the image like a topographic map.
            #Inverted distance (-distance) makes peaks into valleys.
            #The markers act as "seeds", and watershed "floods" each region.
            #The result is a segmented image, where each cell gets a unique label.
            labels_watershed = watershed(-distance, markers, mask=binary_A)
    
            # Visualization of watershed result
            plt.figure(figsize=(8, 8))
            plt.imshow(labels_watershed, cmap='nipy_spectral')
            plt.title('Watershed Segmentation')
            plt.axis('off')
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            #plt.show()
            
            # Save the segmented binary image
            segmented_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented.png")
            cv2.imwrite(segmented_image_path, labels_watershed)
            print(f"Saved segmented image for {bf_file} to {segmented_image_path}")
            all_output_files.append(segmented_image_path)

            # Initialize an empty binary image of the same size as binary_A
            filtered_binary_A = np.zeros_like(binary_A)
    
            # Iterate through each detected region in binary_A
            for prop in region_props_A:
                if prop.area > 0:# Ensure the region has a valid area
                    min_row, min_col, max_row, max_col = prop.bbox
                    # Assign the corresponding segmented region from labels_watershed to filtered_binary_A
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (labels_watershed[min_row:max_row, min_col:max_col] == prop.label)
            
            # Convert filtered_binary_A into a binary uint8 format (0 and 255 values)
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255  # Convert back to uint8
    
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
            #region_area_df.to_excel(region_area_excel_path, index=False)

            print(f"Saved region areas for {bf_file} to {region_area_excel_path}")
    
            plt.figure(figsize=(8, 6))
            plt.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (A) = {threshold:.2f}')
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
    
            # Save the histogram image
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            print(f"Saved histogram for {bf_file} to {segmented_image_path}")
            all_output_files.append(hist_cells_image_path)
    
            # Convert BF image to grayscale and enhance contrast
            grayB = rgb2gray(imageB)
            
            grayB = exposure.equalize_adapthist(grayB)

            # Apply bilateral filter to reduce noise
            grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)

            # Calculate dynamic threshold
            mean_intensity = np.mean(grayB)
            std_intensity = np.std(grayB)
            
            #ORIGINAL WITH VALUE 4
            dynamic_threshold = mean_intensity + 4 * std_intensity
      
            # Apply dynamic threshold
            binary_B = (grayB > dynamic_threshold).astype(np.uint8)
            
            plt.figure(figsize=(8, 6))
            plt.hist(grayB.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.axvline(dynamic_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (B) = {dynamic_threshold:.2f}')
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
    
            # Save the histogram image
            hist_crystals_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_crystals.png")
            plt.savefig(hist_crystals_image_path, dpi=300, bbox_inches='tight')
            #plt.show()
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            print(f"Saved histogram for {bf_file} to {segmented_image_path}")
            all_output_files.append(hist_crystals_image_path)
    
            QApplication.processEvents()  # Refresh PyQt GUI
    
            # Resize for alignment
            filtered_binary_A_resized = cv2.resize(binary_A, (2048, 2048), interpolation=cv2.INTER_AREA)
            binary_B_resized = cv2.resize(binary_B, (2048, 2048), interpolation=cv2.INTER_AREA)

            # Overlap calculation
            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255
    
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

            # Save region-to-cell mapping as CSV
            df_mapp = pd.DataFrame(region_to_cell_mapping)
            df_mapping = df_mapp[(df_mapp["Region_Area (µm²)"] < 10) & (df_mapp["Overlap (pixels)"] > 0)]

            # --- Properly count how many crystals are mapped to each cell ---
            df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())

            # --- Add total number of distinct cells ---
            total_distinct_cells = df_mapping["Associated_Cell"].nunique()
            df_mapping["Total_Cells_with_crystals"] = total_distinct_cells
            total_area = df_mapping["Region_Area (µm²)"].sum()
            total_row = ["","","","Total Area Crystals", total_area,"",""]
            
            # Insert the total row at the end with index "Total"
            df_mapping.loc["Total"] = total_row
            
            cell_to_crystals = defaultdict(list)
            
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
            #df_mapping.to_excel(mapping_excel_path, index=False)

            # --- Merge with region area data ---
            merged_df = df_mapping.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")

            grouped_xlsx_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_All_Datasets.xlsx")

            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)
                df_mapping.to_excel(writer, sheet_name='Crystals', index=False)
                merged_df.to_excel(writer, sheet_name='Cells + crystals', index=False)
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
            #plt.show()
    
            # Save annotated image
            annotated_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)

            print(f"Saved results for {bf_file} to {self.output_folder}")    
            
            all_output_files.append(annotated_image_path)
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

    def start_processing_2(self):
        self.processing_active = True
        self.stop_event.clear()  # Reset stop event
        
        if not self.bf_folder or not self.pl_folder or not self.output_folder:
            self.log("Please select all folders before starting.")
            return
        
        self.log("Starting batch processing...")
        # Define image scale: 7.0917 pixels = 1 µm
        #pixel_to_um = 1 / 7.0917
        try:
            distance_in_px = float(self.pixel_distance_input.text())
            known_um = float(self.known_um_input.text())
            if distance_in_px == 0:
                raise ValueError("Distance in pixels cannot be zero.")
            pixel_to_um = 1/(known_um / distance_in_px)
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

            # Ensure binary_A is the correct shape (resize if necessary)
            if binary_A.shape != grayA.shape:
                binary_A = resize(binary_A, grayA.shape, order=0, preserve_range=True, anti_aliasing=False)

    
            # Apply the Euclidean Distance Transform for watershed segmentation and helps identify the centers of objects(cells).
            distance = distance_transform_edt(binary_A)

            # Find local maxima in the distance image to use as markers, these peaks correspond to cell centers.
            local_maxi = peak_local_max(distance, labels=binary_A, min_distance=1)

            # Label the local maxima,in other words,labels each detected peak with a unique number.
            #These markers will be used as starting points for the watershed algorithm.
            # Initialize markers array (same shape as the image)
            markers = np.zeros_like(distance, dtype=int)
            #markers[local_maxi] = np.arange(1, np.sum(local_maxi) + 1)  # Label the markers

            #markers = np.zeros_like(distance, dtype=int)

            # Assign unique markers to the local maxima positions
            for i, (row, col) in enumerate(local_maxi):
                markers[row, col] = i + 1  # Start from 1

            # Apply watershed
            #Watershed treats the image like a topographic map.
            #Inverted distance (-distance) makes peaks into valleys.
            #The markers act as "seeds", and watershed "floods" each region.
            #The result is a segmented image, where each cell gets a unique label.
            labels_watershed = watershed(-distance, markers, mask=binary_A)
    
            # Visualization of watershed result
            plt.figure(figsize=(8, 8))
            plt.imshow(labels_watershed, cmap='nipy_spectral')
            plt.title('Watershed Segmentation')
            plt.axis('off')
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            #plt.show()
            
            # Save the segmented binary image
            segmented_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Segmented.png")
            cv2.imwrite(segmented_image_path, labels_watershed)
            print(f"Saved segmented image for {bf_file} to {segmented_image_path}")
            all_output_files.append(segmented_image_path)

            # Initialize an empty binary image of the same size as binary_A
            filtered_binary_A = np.zeros_like(binary_A)
    
            # Iterate through each detected region in binary_A
            for prop in region_props_A:
                if prop.area > 0:# Ensure the region has a valid area
                    min_row, min_col, max_row, max_col = prop.bbox
                    # Assign the corresponding segmented region from labels_watershed to filtered_binary_A
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (labels_watershed[min_row:max_row, min_col:max_col] == prop.label)
            
            # Convert filtered_binary_A into a binary uint8 format (0 and 255 values)
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255  # Convert back to uint8
    
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
            #region_area_df.to_excel(region_area_excel_path, index=False)

            print(f"Saved region areas for {bf_file} to {region_area_excel_path}")
    
            plt.figure(figsize=(8, 6))
            plt.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (A) = {threshold:.2f}')
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
    
            # Save the histogram image
            hist_cells_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_cells.png")
            plt.savefig(hist_cells_image_path, dpi=300, bbox_inches='tight')
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            print(f"Saved histogram for {bf_file} to {segmented_image_path}")
            all_output_files.append(hist_cells_image_path)
    
            # Convert BF image to grayscale and enhance contrast
            grayB = rgb2gray(imageB)
            
            grayB = exposure.equalize_adapthist(grayB)

            # Apply bilateral filter to reduce noise
            grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)

            # Calculate dynamic threshold
            mean_intensity = np.mean(grayB)
            std_intensity = np.std(grayB)
            
            #ORIGINAL WITH VALUE 4
            dynamic_threshold = mean_intensity + 4.75 * std_intensity
      
            # Apply dynamic threshold
            binary_B = (grayB > dynamic_threshold).astype(np.uint8)

            binary_B = opening(binary_B)# Remove small noise
            binary_B= morphology.dilation(binary_B, morphology.disk(4)) # Dilation
            binary_B = morphology.closing(binary_B, morphology.disk(4)) # Closing
            binary_B = (binary_B > 0).astype(np.uint8) * 255 # Convert back to binary
            
            plt.figure(figsize=(8, 6))
            plt.hist(grayB.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            plt.axvline(dynamic_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (B) = {dynamic_threshold:.2f}')
            plt.title('Histogram of Pixel Intensities')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
    
            # Save the histogram image
            hist_crystals_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Histogram_crystals.png")
            plt.savefig(hist_crystals_image_path, dpi=300, bbox_inches='tight')
            #plt.show()
            plt.pause(0.001)
            QApplication.processEvents()  # Refresh PyQt GUI
            print(f"Saved histogram for {bf_file} to {segmented_image_path}")
            all_output_files.append(hist_crystals_image_path)
    
            QApplication.processEvents()  # Refresh PyQt GUI
    
            # Resize for alignment
            filtered_binary_A_resized = cv2.resize(binary_A, (2048, 2048), interpolation=cv2.INTER_AREA)
            binary_B_resized = cv2.resize(binary_B, (2048, 2048), interpolation=cv2.INTER_AREA)

            # Overlap calculation
            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255
    
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

            # Save region-to-cell mapping as CSV
            df_mapp = pd.DataFrame(region_to_cell_mapping)
            df_mapping = df_mapp[(df_mapp["Region_Area (µm²)"] < 5) & (df_mapp["Overlap (pixels)"] > 0)]

            # --- Properly count how many crystals are mapped to each cell ---
            df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())

            # --- Add total number of distinct cells ---
            total_distinct_cells = df_mapping["Associated_Cell"].nunique()
            df_mapping["Total_Cells_with_crystals"] = total_distinct_cells
            total_area = df_mapping["Region_Area (µm²)"].sum()
            total_row = ["","","","Total Area Crystals", total_area,"",""]
            
            # Insert the total row at the end with index "Total"
            df_mapping.loc["Total"] = total_row
            
            cell_to_crystals = defaultdict(list)
            
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
            #df_mapping.to_excel(mapping_excel_path, index=False)

            # --- Merge with region area data ---
            merged_df = df_mapping.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")
#-----------------------------------------------------------------------------------------
            # Initialize the column with NaNs
            merged_df["Crystal/Cell Area (%)"] = pd.NA

            # Calculate percentage only for rows except the last two
            merged_df.loc[:-3, "Crystal/Cell Area (%)"] = (
                merged_df.loc[:-3, "Region_Area (µm²)_x"] / merged_df.loc[:-3, "Region_Area (µm²)_y"] * 100
            )
#-------------------------------------------

            grouped_xlsx_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_All_Datasets.xlsx")

            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)
                df_mapping.to_excel(writer, sheet_name='Crystals', index=False)
                merged_df.to_excel(writer, sheet_name='Cells + crystals', index=False)
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
            #plt.show()
    
            # Save annotated image
            annotated_image_path = os.path.join(self.output_folder, f"{os.path.splitext(bf_file)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)

            print(f"Saved results for {bf_file} to {self.output_folder}")    
            
            all_output_files.append(annotated_image_path)
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





# In[ ]:




