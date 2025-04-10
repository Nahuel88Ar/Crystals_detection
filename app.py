#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import streamlit as st
import cv2
import sys
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import tempfile
#matplotlib.use("Qt5Agg")
matplotlib.use("Agg")

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
#from PyQt5.QtWidgets import QApplication
#QApplication.processEvents()

# Conversion factor
PIXEL_TO_UM = 1 / 7.0917

st.title("Batch Image Processing App (Streamlit Version)")

# File uploader for BF and PL folders
bf_files = st.file_uploader("Upload BF images (.tif)", type=["tif"], accept_multiple_files=True)
pl_files = st.file_uploader("Upload PL images (.tif)", type=["tif"], accept_multiple_files=True)

output_dir = st.text_input("Output folder (locally)", value="outputs")
os.makedirs(output_dir, exist_ok=True)

if st.button("Start Processing") and bf_files and pl_files:
    st.write("Starting batch processing...")
    if len(bf_files) != len(pl_files):
        st.error("Mismatch in the number of BF and PL files.")
    else:
        for bf_file, pl_file in zip(bf_files, pl_files):
            st.write(f"Processing {bf_file.name} and {pl_file.name}...")

            # Save files to a temp folder to work with OpenCV
            with tempfile.NamedTemporaryFile(delete=False) as bf_temp, tempfile.NamedTemporaryFile(delete=False) as pl_temp:
                bf_temp.write(bf_file.read())
                pl_temp.write(pl_file.read())
                bf_path = bf_temp.name
                pl_path = pl_temp.name

            imageA = cv2.imread(bf_path)
            imageB = cv2.imread(pl_path)

            if imageA is None or imageB is None:
                st.warning(f"Unable to read {bf_file.name} or {pl_file.name}. Skipping...")
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
            binary_A = opening(binary_A)# Remove small noise
            binary_A = remove_small_objects(binary_A.astype(bool), min_size=500)# Remove small objects
            binary_A = remove_small_holes(binary_A, area_threshold=100000)# Fill small holes
            binary_A = morphology.dilation(binary_A, morphology.disk(4)) # Dilation
            binary_A = morphology.closing(binary_A, morphology.disk(4)) # Closing
            binary_A = (binary_A > 0).astype(np.uint8) * 255 # Convert back to binary
    
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
            
            # Further processing and saving
            result_path = os.path.join(output_dir, f"{bf_file.name}_Segmented.png")
            cv2.imwrite(result_path, labels_watershed)
            st.image(result_path, caption=f"Segmented {bf_file.name}", use_column_width=True)

            # Initialize an empty binary image of the same size as binary_A
            filtered_binary_A = np.zeros_like(binary_A)

            # Iterate through each detected region in binary_A
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (labels_watershed[min_row:max_row, min_col:max_col] == prop.label)

            # Convert filtered_binary_A into binary uint8 format
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255

            # Create a DataFrame for the regions
            region_area = pd.DataFrame({
                "Region_Label": [region.label for region in region_props_A],
                "Region_Area (pixels)": [region.area for region in region_props_A],
                "Region_Area (µm²)": [region.area * (PIXEL_TO_UM ** 2) for region in region_props_A]
            })

            # Filter regions larger than 0 µm²
            region_area_df = region_area[region_area["Region_Area (µm²)"] > 0]

            # Calculate total area
            total_area = region_area_df["Region_Area (µm²)"].sum()

            # Add total to the DataFrame
            region_area_df.loc["Total"] = ["Total", "", total_area]

            # Save to Excel
            excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Area_in_um2.xlsx")
            region_area_df.to_excel(excel_path, index=False)

            st.success(f"Saved region areas for {bf_file.name} to Excel")

            with open(excel_path, "rb") as f:
                st.download_button("Download Excel file", f, file_name=os.path.basename(excel_path))

            # Create and display histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (A) = {threshold:.2f}')
            ax.set_title('Histogram of Pixel Intensities')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.legend()

            # Save histogram
            histogram_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_cells.png")
            plt.savefig(histogram_path, dpi=300, bbox_inches='tight')

            # Show in Streamlit
            st.pyplot(fig)

            with open(histogram_path, "rb") as f:
                st.download_button("Download Histogram", f, file_name=os.path.basename(histogram_path))
                
            # Convert BF image to grayscale and enhance contrast
            grayB = rgb2gray(imageB)
            grayB = exposure.equalize_adapthist(grayB)

            # Apply bilateral filter to reduce noise
            grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)

            # Calculate dynamic threshold
            mean_intensity = np.mean(grayB)
            std_intensity = np.std(grayB)
            dynamic_threshold = mean_intensity + 4 * std_intensity

            # Apply dynamic threshold
            binary_B = (grayB > dynamic_threshold).astype(np.uint8)

            # Create a histogram to visualize thresholding
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(grayB.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            ax.axvline(dynamic_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (B) = {dynamic_threshold:.2f}')
            ax.set_title('Histogram of Pixel Intensities')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.legend()

            # Save histogram image
            histogram_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_crystals.png")
            plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
            st.image(histogram_path, caption=f"Histogram for {bf_file.name}", use_column_width=True)

            # Plot binary image for 'B' and save it
            binaryB_image_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_binaryB.png")
            cv2.imwrite(binaryB_image_path, binary_B)
            st.image(binaryB_image_path, caption=f"Binary Image for {bf_file.name}", use_column_width=True)

            # Resize for alignment and calculate overlap
            filtered_binary_A_resized = cv2.resize(binary_B, (2048, 2048), interpolation=cv2.INTER_AREA)
            binary_B_resized = cv2.resize(binary_B, (2048, 2048), interpolation=cv2.INTER_AREA)
            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255

            # Save overlap image
            overlap_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Overlap.png")
            cv2.imwrite(overlap_path, overlap)
            st.image(overlap_path, caption=f"Overlap Image for {bf_file.name}", use_column_width=True)

            # Clustering and region-cell mapping
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
                    "Region_Area (µm²)": region.area * (PIXEL_TO_UM ** 2)
                })

            # Convert to DataFrame and save to Excel
            df_mapping = pd.DataFrame(region_to_cell_mapping)
            df_mapping = df_mapping[df_mapping["Region_Area (µm²)"] > 0]

            # Save Excel with mapping
            mapping_excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Cell_Mapping.xlsx")
            df_mapping.to_excel(mapping_excel_path, index=False)

            with open(mapping_excel_path, "rb") as f:
                st.download_button("Download Region-Cell Mapping", f, file_name=os.path.basename(mapping_excel_path))

            st.success(f"Saved region-to-cell mapping for {bf_file.name} to Excel")
            
            # Perform region-to-cell mapping (assuming the previous parts are already processed)
            region_to_cell_mapping = []
            cell_labels = label(imageA)  # Replace with your actual segmentation labels
            cell_props = regionprops(cell_labels)
            region_labels = label(imageB)  # Replace with actual overlap or region labels
            region_props = regionprops(region_labels)

            # Annotate the image with region and cell information
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

            # Create and display the image with annotations and coincidences
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Show annotated image (detections)
            ax[0].imshow(annotated_image, cmap='gray')
            ax[0].set_title('Detections')
            ax[0].axis('off')  # Hide axes

            # Show overlap image (coincidences)
            overlap = (np.logical_and(imageA > 0, imageB > 0)).astype(np.uint8) * 255
            ax[1].imshow(overlap, cmap='gray')
            ax[1].set_title('Coincidences')
            ax[1].axis('off')  # Hide axes

            plt.tight_layout()
            st.pyplot(fig)  # Display the plot in Streamlit

            # Save annotated image and provide download button
            annotated_image_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)

            with open(annotated_image_path, "rb") as f:
                st.download_button("Download Annotated Image", f, file_name=os.path.basename(annotated_image_path))

            st.success(f"Saved annotated image for {bf_file.name} to {output_dir}")
        st.success("Processing complete!")

