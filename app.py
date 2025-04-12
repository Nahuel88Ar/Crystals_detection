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
            #---------------NEW
            df_mapp = pd.DataFrame(region_to_cell_mapping)
            
            # First, drop any rows with NaN in Associated_Cell to avoid counting issues
            df_mapping = df_mapp[df_mapp["Region_Area (µm²)"] > 0].copy()
            df_mapping = df_mapping[df_mapping["Associated_Cell"].notnull()]
            
            # Ensure Associated_Cell is of type int (in case it's float due to NaNs)
            df_mapping["Associated_Cell"] = df_mapping["Associated_Cell"].astype(int)

            # Now compute the count
            associated_cell_counts = df_mapping["Associated_Cell"].value_counts()

            # Map it back to the DataFrame
            df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(associated_cell_counts)

            # Add total distinct cells
            df_mapping["Total_Distinct_Cells"] = df_mapping["Associated_Cell"].nunique()

            # Total row
            df_mapping.loc["Total", "Region_Area (µm²)"] = df_mapping["Region_Area (µm²)"].sum()

            mapping_excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Cell_Mapping.xlsx")
            df_mapping.to_excel(mapping_excel_path, index=False)
            
            # Group by Associated_Cell and count regions, then calculate percentages
            cell_grouped_df = pd.DataFrame(region_to_cell_mapping)
            cell_region_count = cell_grouped_df.groupby("Associated_Cell")["Region_Label"].count().reset_index()
            cell_region_count.columns = ["Associated_Cell", "Region_Count"]
            total_region_count = cell_region_count["Region_Count"].sum()
            cell_region_count["Percentage"] = (cell_region_count["Region_Count"] / total_region_count) * 100

            # Remove columns "D" and "E" from the first sheet (final grouped DataFrame)
            final_grouped_df = cell_region_count.drop(columns=["Percentage"])
    
            # Create a new DataFrame to count how many times each Associated_Cell has the same value as Region_Label
            region_to_associated_cell_mapping = []
            for region in region_to_cell_mapping:
                region_label = region["Region_Label"]
                associated_cell = region["Associated_Cell"]
                # Check if the Associated Cell matches the Region Label
                region_to_associated_cell_mapping.append({
                    "Region_Label": region_label,
                    "Associated_Cell": associated_cell
                })

            # Convert to DataFrame
            region_association_df = pd.DataFrame(region_to_associated_cell_mapping)

            # Count how many times each Associated_Cell has the same value as the Region_Label
            matching_cell_count = region_association_df[region_association_df["Region_Label"] == region_association_df["Associated_Cell"]].groupby("Region_Label").size().reset_index(name="Matching_Associated_Cell_Count")

            # Merge the new data with the original cell_region_count dataframe
            final_grouped_df = pd.merge(final_grouped_df, matching_cell_count, how="left", left_on="Associated_Cell", right_on="Region_Label")
         
            st.success(f"Saved Crystal dataset for {bf_file.name} to Excel")
            
            #NEW
            # Drop any rows where Region_Label is not a digit (e.g., "Total")
            region_area_df = region_area_df[region_area_df["Region_Label"].apply(lambda x: str(x).isdigit())].copy()

            region_area_df["Region_Label"] = region_area_df["Region_Label"].astype(int)
            
            merged_df = df_mapping.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")
            
            grouped_xlsx_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_All_Datasets.xlsx")
            
            # Saving to an Excel file with multiple sheets
            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                # Save the DataFrame (region_area_df) to the first sheet
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)  # Shortened sheet name
                
                # Save the DataFrame (df_mapping) to the second sheet
                df_mapping.to_excel(writer, sheet_name='Crystals', index=False)  # Shortened sheet name
                
                # Save the DataFrame (merged_df) to the thitd sheet
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
                    "Region_Area (µm²)": region.area * (PIXEL_TO_UM ** 2)
                })

            # Create and clean DataFrame
            df_mapping = pd.DataFrame(region_to_cell_mapping)
            df_mapping = df_mapping[df_mapping["Region_Area (µm²)"] > 0].copy()  # Avoid SettingWithCopyWarning

            # Ensure Associated_Cell is int to enable proper counting
            df_mapping["Associated_Cell"] = df_mapping["Associated_Cell"].astype(int)

            # Add per-cell occurrence count
            df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())

            # Calculate total distinct cells
            total_distinct_cells = df_mapping["Associated_Cell"].nunique()

            # Optional: add Total row separately for visual summary
            total_row = pd.DataFrame([{
                "Region_Label": "Total",
                "Associated_Cell": "",
                "Overlap (pixels)": "",
                "Region_Area (pixels)": "",
                "Region_Area (µm²)": df_mapping["Region_Area (µm²)"].sum(),
                "Associated_Cell_Count": "",
                "Total_Distinct_Cells": total_distinct_cells
            }])

            df_mapping_with_total = pd.concat([df_mapping, total_row], ignore_index=True)

            # Save Excel
            mapping_excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Cell_Mapping.xlsx")
            df_mapping_with_total.to_excel(mapping_excel_path, index=False)
            
            # Group by Associated_Cell and count regions, then calculate percentages
            cell_grouped_df = pd.DataFrame(region_to_cell_mapping)
            cell_region_count = cell_grouped_df.groupby("Associated_Cell")["Region_Label"].count().reset_index()
            cell_region_count.columns = ["Associated_Cell", "Region_Count"]
            total_region_count = cell_region_count["Region_Count"].sum()
            cell_region_count["Percentage"] = (cell_region_count["Region_Count"] / total_region_count) * 100

            # Remove columns "D" and "E" from the first sheet (final grouped DataFrame)
            final_grouped_df = cell_region_count.drop(columns=["Percentage"])
    
            # Create a new DataFrame to count how many times each Associated_Cell has the same value as Region_Label
            region_to_associated_cell_mapping = []
            for region in region_to_cell_mapping:
                region_label = region["Region_Label"]
                associated_cell = region["Associated_Cell"]
                # Check if the Associated Cell matches the Region Label
                region_to_associated_cell_mapping.append({
                    "Region_Label": region_label,
                    "Associated_Cell": associated_cell
                })

            # Convert to DataFrame
            region_association_df = pd.DataFrame(region_to_associated_cell_mapping)

            # Count how many times each Associated_Cell has the same value as the Region_Label
            matching_cell_count = region_association_df[region_association_df["Region_Label"] == region_association_df["Associated_Cell"]].groupby("Region_Label").size().reset_index(name="Matching_Associated_Cell_Count")

            # Merge the new data with the original cell_region_count dataframe
            final_grouped_df = pd.merge(final_grouped_df, matching_cell_count, how="left", left_on="Associated_Cell", right_on="Region_Label")

            st.success(f"Saved Crystal dataset for {bf_file.name} to Excel")
            
            merged_df = df_mapping_with_total.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")
            
            grouped_xlsx_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_All_Datasets.xlsx")
            
            # Saving to an Excel file with multiple sheets
            with pd.ExcelWriter(grouped_xlsx_path, engine='xlsxwriter') as writer:
                # Save the DataFrame (region_area_df) to the first sheet
                region_area_df.to_excel(writer, sheet_name='Cells', index=False)  # Shortened sheet name
                
                # Save the DataFrame (df_mapping) to the second sheet
                df_mapping_with_total.to_excel(writer, sheet_name='Crystals', index=False)  # Shortened sheet name
                
                # Save the DataFrame (merged_df) to the thitd sheet
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

