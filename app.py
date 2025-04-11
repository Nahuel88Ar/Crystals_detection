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
matplotlib.use("Agg")  # Headless-safe backend

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

from scipy.ndimage import distance_transform_edt, label as ndi_label
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from scipy.signal import find_peaks

from xlsxwriter import Workbook

# Conversion factor
PIXEL_TO_UM = 1 / 7.0917

st.title("Batch Image Processing App (Streamlit Version)")

# File uploader for BF and PL folders
bf_files = st.file_uploader("Upload BF images (.tif)", type=["tif"], accept_multiple_files=True)
pl_files = st.file_uploader("Upload PL images (.tif)", type=["tif"], accept_multiple_files=True)

output_dir = st.text_input("Output folder (locally)", value="outputs")
os.makedirs(output_dir, exist_ok=True)

if st.button("Start script 1") and bf_files and pl_files:
    st.write("Starting batch processing...")
    if len(bf_files) != len(pl_files):
        st.error("Mismatch in the number of BF and PL files.")
    else:
        for bf_file, pl_file in zip(bf_files, pl_files):
            st.write(f"Processing {bf_file.name} and {pl_file.name}...")

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

            grayA = rgb2gray(imageA)
            grayA = exposure.equalize_adapthist(grayA)
            grayA = cv2.bilateralFilter((grayA * 255).astype(np.uint8), 9, 75, 75)
            threshold = threshold_otsu(grayA)
            binary_A = (grayA < threshold).astype(np.uint8) * 255

            binary_A = morphology.opening(binary_A)
            binary_A = morphology.remove_small_objects(binary_A.astype(bool), min_size=500)
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=100000)
            binary_A = morphology.dilation(binary_A, morphology.disk(4))
            binary_A = morphology.closing(binary_A, morphology.disk(4))
            binary_A = (binary_A > 0).astype(np.uint8) * 255

            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            if binary_A.shape != grayA.shape:
                binary_A = resize(binary_A, grayA.shape, order=0, preserve_range=True, anti_aliasing=False)

            distance = distance_transform_edt(binary_A)
            local_maxi = peak_local_max(distance, labels=binary_A, min_distance=1)
            markers = np.zeros_like(distance, dtype=int)
            for i, (row, col) in enumerate(local_maxi):
                markers[row, col] = i + 1

            labels_watershed = watershed(-distance, markers, mask=binary_A)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(labels_watershed, cmap='nipy_spectral')
            ax.set_title('Watershed Segmentation')
            ax.axis('off')
            st.pyplot(fig)

            result_path = os.path.join(output_dir, f"{bf_file.name}_Segmented.png")
            cv2.imwrite(result_path, labels_watershed)
            st.image(result_path, caption=f"Segmented {bf_file.name}", use_container_width=True)

            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (labels_watershed[min_row:max_row, min_col:max_col] == prop.label)
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255

            region_area = pd.DataFrame({
                "Region_Label": [region.label for region in region_props_A],
                "Region_Area (pixels)": [region.area for region in region_props_A],
                "Region_Area (µm²)": [region.area * (PIXEL_TO_UM ** 2) for region in region_props_A]
            })
            region_area_df = region_area[region_area["Region_Area (µm²)"] > 0]
            total_area = region_area_df["Region_Area (µm²)"].sum()
            region_area_df.loc["Total"] = ["Total", "", total_area]

            excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Area_in_um2.xlsx")
            region_area_df.to_excel(excel_path, index=False)

            st.success(f"Saved cells area for {bf_file.name} to Excel")
            #with open(excel_path, "rb") as f:
            #    st.download_button("Download dataset of cells", f, file_name=os.path.basename(excel_path))

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (A) = {threshold:.2f}')
            ax.set_title('Histogram of Pixel Intensities')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)

            grayB = rgb2gray(imageB)
            grayB = exposure.equalize_adapthist(grayB)
            grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)
            mean_intensity = np.mean(grayB)
            std_intensity = np.std(grayB)
            dynamic_threshold = mean_intensity + 4 * std_intensity
            binary_B = (grayB > dynamic_threshold).astype(np.uint8)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(grayB.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            ax.axvline(dynamic_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (B) = {dynamic_threshold:.2f}')
            ax.set_title('Histogram of Pixel Intensities')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)

            filtered_binary_A_resized = cv2.resize(binary_B, (2048, 2048), interpolation=cv2.INTER_AREA)
            binary_B_resized = cv2.resize(binary_B, (2048, 2048), interpolation=cv2.INTER_AREA)
            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cv2.cvtColor(overlap, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Overlap Image for {bf_file.name}")
            ax.axis('off')
            st.pyplot(fig)

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

            df_mapping = pd.DataFrame(region_to_cell_mapping)
            df_mapping = df_mapping[df_mapping["Region_Area (µm²)"] > 0]

            # Add additional stats to the DataFrame
            df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())
            total_distinct_cells = df_mapping["Associated_Cell"].nunique()
            df_mapping["Total_Distinct_Cells"] = total_distinct_cells
            df_mapping.loc["Total", "Region_Area (µm²)"] = df_mapping["Region_Area (µm²)"].sum()

            mapping_excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Cell_Mapping.xlsx")
            df_mapping.to_excel(mapping_excel_path, index=False)

            #with open(mapping_excel_path, "rb") as g:
            #    st.download_button("Download Crystal dataset", g, file_name=os.path.basename(mapping_excel_path))

            st.success(f"Saved Crystal dataset for {bf_file.name} to Excel")
            
            merged_df = df_mapping.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")

            grouped_xlsx_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_All_Datasets.xlsx")
            
            # Saving to an Excel file with multiple sheets
            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                # Save the original DataFrame (final_grouped_df) to the first sheet
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)  # Shortened sheet name
                
                # Save the original DataFrame (final_grouped_df) to the first sheet
                merged_df.to_excel(writer, sheet_name='Cells + crystals', index=False)  # Shortened sheet name

            st.success(f"Saved all datasets for {bf_file.name} to Excel")
    
            with open(grouped_xlsx_path, "rb") as g:
                st.download_button("Download all datasets", g, file_name=os.path.basename(grouped_xlsx_path))

            annotated_image = imageA.copy()
            for mapping in region_to_cell_mapping:
                region_label = mapping["Region_Label"]
                associated_cell = mapping["Associated_Cell"]
                if associated_cell:
                    region = next(r for r in region_props if r.label == region_label)
                    min_row, min_col, max_row, max_col = region.bbox
                    cv2.rectangle(annotated_image, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"Cell {associated_cell}", (min_col, min_row - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(annotated_image, cmap='gray')
            ax[0].set_title('Detections')
            ax[0].axis('off')
            ax[1].imshow(overlap, cmap='gray')
            ax[1].set_title('Coincidences')
            ax[1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

            annotated_image_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)
            st.image(annotated_image_path, caption="Annotated Image", use_container_width=True)

            st.success(f"Saved annotated image for {bf_file.name} to {output_dir}")
        st.success("Processing complete!")
        
if st.button("Start script 2") and bf_files and pl_files:
    st.write("Starting batch processing...")
    if len(bf_files) != len(pl_files):
        st.error("Mismatch in the number of BF and PL files.")
    else:
        for bf_file, pl_file in zip(bf_files, pl_files):
            st.write(f"Processing {bf_file.name} and {pl_file.name}...")

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

            grayA = rgb2gray(imageA)
            grayA = exposure.equalize_adapthist(grayA)
            grayA = cv2.bilateralFilter((grayA * 255).astype(np.uint8), 9, 75, 75)
            threshold = threshold_otsu(grayA)
            binary_A = (grayA < threshold).astype(np.uint8) * 255

            binary_A = morphology.opening(binary_A)
            binary_A = morphology.remove_small_objects(binary_A.astype(bool), min_size=500)
            binary_A = morphology.remove_small_holes(binary_A, area_threshold=100000)
            binary_A = morphology.dilation(binary_A, morphology.disk(4))
            binary_A = morphology.closing(binary_A, morphology.disk(4))
            binary_A = (binary_A > 0).astype(np.uint8) * 255

            region_labels_A = label(binary_A)
            region_props_A = regionprops(region_labels_A)

            if binary_A.shape != grayA.shape:
                binary_A = resize(binary_A, grayA.shape, order=0, preserve_range=True, anti_aliasing=False)

            distance = distance_transform_edt(binary_A)
            local_maxi = peak_local_max(distance, labels=binary_A, min_distance=1)
            markers = np.zeros_like(distance, dtype=int)
            for i, (row, col) in enumerate(local_maxi):
                markers[row, col] = i + 1

            labels_watershed = watershed(-distance, markers, mask=binary_A)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(labels_watershed, cmap='nipy_spectral')
            ax.set_title('Watershed Segmentation')
            ax.axis('off')
            st.pyplot(fig)

            result_path = os.path.join(output_dir, f"{bf_file.name}_Segmented.png")
            cv2.imwrite(result_path, labels_watershed)
            st.image(result_path, caption=f"Segmented {bf_file.name}", use_container_width=True)

            filtered_binary_A = np.zeros_like(binary_A)
            for prop in region_props_A:
                if prop.area > 0:
                    min_row, min_col, max_row, max_col = prop.bbox
                    filtered_binary_A[min_row:max_row, min_col:max_col] = (labels_watershed[min_row:max_row, min_col:max_col] == prop.label)
            filtered_binary_A = (filtered_binary_A > 0).astype(np.uint8) * 255

            region_area = pd.DataFrame({
                "Region_Label": [region.label for region in region_props_A],
                "Region_Area (pixels)": [region.area for region in region_props_A],
                "Region_Area (µm²)": [region.area * (PIXEL_TO_UM ** 2) for region in region_props_A]
            })
            region_area_df = region_area[region_area["Region_Area (µm²)"] > 0]
            total_area = region_area_df["Region_Area (µm²)"].sum()
            region_area_df.loc["Total"] = ["Total", "", total_area]

            excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Area_in_um2.xlsx")
            region_area_df.to_excel(excel_path, index=False)

            st.success(f"Saved cells area for {bf_file.name} to Excel")
            
            #with open(excel_path, "rb") as f:
            #    st.download_button("Download dataset of cells", f, file_name=os.path.basename(excel_path))

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(grayA.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (A) = {threshold:.2f}')
            ax.set_title('Histogram of Pixel Intensities')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)

            grayB = rgb2gray(imageB)
            grayB = exposure.equalize_adapthist(grayB)
            grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)
            mean_intensity = np.mean(grayB)
            std_intensity = np.std(grayB)
            dynamic_threshold = mean_intensity + 5 * std_intensity
            binary_B = (grayB > dynamic_threshold).astype(np.uint8)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(grayB.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
            ax.axvline(dynamic_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (B) = {dynamic_threshold:.2f}')
            ax.set_title('Histogram of Pixel Intensities')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)

            filtered_binary_A_resized = cv2.resize(binary_B, (2048, 2048), interpolation=cv2.INTER_AREA)
            binary_B_resized = cv2.resize(binary_B, (2048, 2048), interpolation=cv2.INTER_AREA)
            overlap = (np.logical_and(filtered_binary_A_resized > 0, binary_B_resized > 0)).astype(np.uint8) * 255

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cv2.cvtColor(overlap, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Overlap Image for {bf_file.name}")
            ax.axis('off')
            st.pyplot(fig)

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

            df_mapping = pd.DataFrame(region_to_cell_mapping)
            df_mapping = df_mapping[df_mapping["Region_Area (µm²)"] > 0]

            # Add additional stats to the DataFrame
            df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())
            total_distinct_cells = df_mapping["Associated_Cell"].nunique()
            df_mapping["Total_Distinct_Cells"] = total_distinct_cells
            df_mapping.loc["Total", "Region_Area (µm²)"] = df_mapping["Region_Area (µm²)"].sum()

            mapping_excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Cell_Mapping.xlsx")
            df_mapping.to_excel(mapping_excel_path, index=False)

            #with open(mapping_excel_path, "rb") as g:
            #    st.download_button("Download Crystal dataset", g, file_name=os.path.basename(mapping_excel_path))

            st.success(f"Saved Crystal dataset for {bf_file.name} to Excel")
            
            merged_df = df_mapping.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")

            grouped_xlsx_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_All_Datasets.xlsx")
            
            # Saving to an Excel file with multiple sheets
            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                # Save the original DataFrame (final_grouped_df) to the first sheet
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)  # Shortened sheet name
                
                # Save the original DataFrame (final_grouped_df) to the first sheet
                merged_df.to_excel(writer, sheet_name='Cells + crystals', index=False)  # Shortened sheet name

            st.success(f"Saved all datasets for {bf_file.name} to Excel")
            
            with open(grouped_xlsx_path, "rb") as g:
                st.download_button("Download all datasets", g, file_name=os.path.basename(grouped_xlsx_path))

            annotated_image = imageA.copy()
            for mapping in region_to_cell_mapping:
                region_label = mapping["Region_Label"]
                associated_cell = mapping["Associated_Cell"]
                if associated_cell:
                    region = next(r for r in region_props if r.label == region_label)
                    min_row, min_col, max_row, max_col = region.bbox
                    cv2.rectangle(annotated_image, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"Cell {associated_cell}", (min_col, min_row - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(annotated_image, cmap='gray')
            ax[0].set_title('Detections')
            ax[0].axis('off')
            ax[1].imshow(overlap, cmap='gray')
            ax[1].set_title('Coincidences')
            ax[1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

            annotated_image_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Annotated_Image_with_Clustering.png")
            cv2.imwrite(annotated_image_path, annotated_image)
            st.image(annotated_image_path, caption="Annotated Image", use_container_width=True)

            st.success(f"Saved annotated image for {bf_file.name} to {output_dir}")
        st.success("Processing complete!")

