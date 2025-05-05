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

from collections import defaultdict

import zipfile

# === File Inputs ===
bf_files = st.file_uploader("Upload BF files", type=["png", "jpg","tif"], accept_multiple_files=True)
pl_files = st.file_uploader("Upload PL files", type=["png", "jpg","tif"], accept_multiple_files=True)

# Example usage
if bf_files and pl_files:
    st.success(f"Found {len(bf_files)} BF files and {len(pl_files)} PL files.")
    # You can now loop through them for processing
    for bf, pl in zip(bf_files, pl_files):
        st.write(f"Processing: {bf.name} and {pl.name}")
        
output_dir = "outputs"
PIXEL_TO_UM = 1 / 7.0917  # Example pixel-to-micron conversion
os.makedirs(output_dir, exist_ok=True)

# Session State Initialization
if "script1_done" not in st.session_state:
    st.session_state.script1_done = False
if "script1_results" not in st.session_state:
    st.session_state.script1_results = []
if "zip_path_1" not in st.session_state:
    st.session_state.zip_path_1 = None

# Start Button
if st.button("Start script 1"):
    if not bf_files or not pl_files:
        st.warning("Please upload both BF and PL files.")
    elif len(bf_files) != len(pl_files):
        st.error("Mismatch in number of BF and PL files.")
    else:
        st.session_state.script1_done = True
        st.session_state.script1_results.clear()

# Processing Logic
if st.session_state.script1_done:
    st.write("🔄 Starting batch processing...")
    all_output_files = []

    for bf_file, pl_file in zip(bf_files, pl_files):
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

        distance = distance_transform_edt(binary_A)
        local_maxi = peak_local_max(distance, labels=binary_A, min_distance=1)
        markers = np.zeros_like(distance, dtype=int)
        for i, (row, col) in enumerate(local_maxi):
            markers[row, col] = i + 1

        labels_watershed = watershed(-distance, markers, mask=binary_A)

        result_path = os.path.join(output_dir, f"{bf_file.name}_Segmented.png")
        cv2.imwrite(result_path, labels_watershed)
        all_output_files.append(result_path)

        region_area_df = pd.DataFrame({
            "Region_Label": [r.label for r in region_props_A],
            "Region_Area (pixels)": [r.area for r in region_props_A],
            "Region_Area (µm²)": [r.area * (PIXEL_TO_UM ** 2) for r in region_props_A]
        })

        region_area_df = region_area_df[region_area_df["Region_Area (µm²)"] > 0]
        region_area_df.loc["Total Area"] = ["", "Total Area", region_area_df["Region_Area (µm²)"].sum()]
        region_area_df.loc["Total Cells"] = ["", "Total Cells", len(region_area_df) - 1]

        excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Area.xlsx")
        region_area_df.to_excel(excel_path, index=False)

        # Histogram A
        fig, ax = plt.subplots()
        ax.hist(grayA.ravel(), bins=256, range=[0, 255])
        ax.axvline(threshold, color='red', linestyle='--')
        hist_path_A = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_A.png")
        fig.savefig(hist_path_A)
        all_output_files.append(hist_path_A)

        # Image B thresholding
        grayB = rgb2gray(imageB)
        grayB = exposure.equalize_adapthist(grayB)
        grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)
        mean_intensity = np.mean(grayB)
        std_intensity = np.std(grayB)
        dynamic_threshold = mean_intensity + 5.5 * std_intensity
        binary_B = (grayB > dynamic_threshold).astype(np.uint8)

        fig, ax = plt.subplots()
        ax.hist(grayB.ravel(), bins=256, range=[0, 255])
        ax.axvline(dynamic_threshold, color='red', linestyle='--')
        hist_path_B = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_B.png")
        fig.savefig(hist_path_B)
        all_output_files.append(hist_path_B)

        overlap = (np.logical_and(cv2.resize(binary_A, (2048, 2048)) > 0, cv2.resize(binary_B, (2048, 2048)) > 0)).astype(np.uint8) * 255
        overlap_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Overlap.png")
        cv2.imwrite(overlap_path, overlap)
        all_output_files.append(overlap_path)

        # Region associations
        region_props = regionprops(label(overlap))
        cell_props = region_props_A
        crystal_to_cell = []
        cell_to_crystals = defaultdict(list)

        for region in region_props:
            region_coords = set(map(tuple, region.coords))
            best_match = None
            max_overlap = 0
            for cell in cell_props:
                cell_coords = set(map(tuple, cell.coords))
                overlap_area = len(region_coords & cell_coords)
                if overlap_area > 0:
                    cell_to_crystals[cell.label].append(region.label)
                if overlap_area > max_overlap:
                    max_overlap = overlap_area
                    best_match = cell.label
            crystal_to_cell.append({
                "Region_Label": region.label,
                "Associated_Cell": best_match,
                "Overlap (pixels)": max_overlap,
                "Region_Area (pixels)": region.area,
                "Region_Area (µm²)": region.area * (PIXEL_TO_UM ** 2)
            })

        df_mapping = pd.DataFrame(crystal_to_cell)
        df_mapping = df_mapping[(df_mapping["Region_Area (µm²)"] < 10) & (df_mapping["Overlap (pixels)"] > 0)]
        df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())
        df_mapping["Total_Cells_with_crystals"] = df_mapping["Associated_Cell"].nunique()
        df_mapping.loc["Total"] = ["", "", "", "Total Area Crystals", df_mapping["Region_Area (µm²)"].sum(), "", ""]

        cell_crystal_df = pd.DataFrame([
            {"Cell_Label": k, "Crystal_Labels": ", ".join(map(str, v)), "Crystal_Count": len(v)}
            for k, v in cell_to_crystals.items()
        ])

        merged_df = df_mapping.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")

        grouped_xlsx_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_All_Datasets.xlsx")
        with pd.ExcelWriter(grouped_xlsx_path, engine="xlsxwriter") as writer:
            region_area_df.to_excel(writer, sheet_name="Cells", index=False)
            df_mapping.to_excel(writer, sheet_name="Crystals", index=False)
            merged_df.to_excel(writer, sheet_name="Cells + Crystals", index=False)
            cell_crystal_df.to_excel(writer, sheet_name="Cell-Crystal Map", index=False)

        # Annotated Image
        annotated_image = cv2.cvtColor(imageA, cv2.COLOR_GRAY2BGR) if imageA.ndim == 2 else imageA.copy()
        for _, mapping in df_mapping.iterrows():
            if pd.notna(mapping["Associated_Cell"]):
                region = next((r for r in region_props if r.label == mapping["Region_Label"]), None)
                if region:
                    min_row, min_col, max_row, max_col = region.bbox
                    cv2.rectangle(annotated_image, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"Cell {int(mapping['Associated_Cell'])}", (min_col, max(min_row - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, lineType=cv2.LINE_AA)

        annotated_image_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Annotated.png")
        cv2.imwrite(annotated_image_path, annotated_image)
        all_output_files.append(annotated_image_path)

        # Save session result
        st.session_state.script1_results.append({
            "bf_name": bf_file.name,
            "excel_path": grouped_xlsx_path,
            "annotated_img_path": annotated_image_path,
            "overlap_path": overlap_path,
            "hist_A_path": hist_path_A,
            "hist_B_path": hist_path_B,
        })

    # Create ZIP
    zip_path_1 = os.path.join(output_dir, "All_Images_histograms.zip")
    with zipfile.ZipFile(zip_path_1, 'w') as zipf_1:
        for file_path in all_output_files:
            zipf_1.write(file_path, arcname=os.path.basename(file_path))
    st.session_state.zip_path_1 = zip_path_1
    st.success("✅ Processing complete!")

# Display Outputs and Download Buttons
if st.session_state.script1_results:
    st.header("📦 Results")

    for result1 in st.session_state.script1_results:
        st.subheader(f"📁 {result1['bf_name']}")
        st.image(result1["annotated_img_path"], caption="Annotated Image")
        st.image(result1["overlap_path"], caption="Overlap Image")

        with open(result1["excel_path"], "rb") as f1:
            st.download_button("📊 Download Dataset", f1, file_name=os.path.basename(result1["excel_path"]),key=f"download_button_{os.path.basename(result1['excel_path'])}")

        #with open(result1["hist_A_path"], "rb") as f1:
        #    st.download_button("📈 Download Histogram A", f1, file_name=os.path.basename(result1["hist_A_path"]))

        #with open(result1["hist_B_path"], "rb") as f1:
        #    st.download_button("📉 Download Histogram B", f1, file_name=os.path.basename(result1["hist_B_path"]))

    with open(st.session_state.zip_path_1, "rb") as zf_1:
        st.download_button("🗂️ Download All Images and Histograms", zf_1, file_name="All_Images_histograms.zip")

#-----------------------------------------------------------------------------------------------------------------------------------

# Session State Initialization
if "script2_done" not in st.session_state:
    st.session_state.script2_done = False
if "script2_results" not in st.session_state:
    st.session_state.script2_results = []
if "zip_path_2" not in st.session_state:
    st.session_state.zip_path_2 = None

# Start Button
if st.button("Start script 2"):
    if not bf_files or not pl_files:
        st.warning("Please upload both BF and PL files.")
    elif len(bf_files) != len(pl_files):
        st.error("Mismatch in number of BF and PL files.")
    else:
        st.session_state.script2_done = True
        st.session_state.script2_results.clear()

# Processing Logic
if st.session_state.script2_done:
    st.write("🔄 Starting batch processing...")
    all_output_files = []

    for bf_file, pl_file in zip(bf_files, pl_files):
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

        distance = distance_transform_edt(binary_A)
        local_maxi = peak_local_max(distance, labels=binary_A, min_distance=1)
        markers = np.zeros_like(distance, dtype=int)
        for i, (row, col) in enumerate(local_maxi):
            markers[row, col] = i + 1

        labels_watershed = watershed(-distance, markers, mask=binary_A)

        result_path = os.path.join(output_dir, f"{bf_file.name}_Segmented.png")
        cv2.imwrite(result_path, labels_watershed)
        all_output_files.append(result_path)

        region_area_df = pd.DataFrame({
            "Region_Label": [r.label for r in region_props_A],
            "Region_Area (pixels)": [r.area for r in region_props_A],
            "Region_Area (µm²)": [r.area * (PIXEL_TO_UM ** 2) for r in region_props_A]
        })

        region_area_df = region_area_df[region_area_df["Region_Area (µm²)"] > 0]
        region_area_df.loc["Total Area"] = ["", "Total Area", region_area_df["Region_Area (µm²)"].sum()]
        region_area_df.loc["Total Cells"] = ["", "Total Cells", len(region_area_df) - 1]

        excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Area.xlsx")
        region_area_df.to_excel(excel_path, index=False)

        # Histogram A
        fig, ax = plt.subplots()
        ax.hist(grayA.ravel(), bins=256, range=[0, 255])
        ax.axvline(threshold, color='red', linestyle='--')
        hist_path_A = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_A.png")
        fig.savefig(hist_path_A)
        all_output_files.append(hist_path_A)

        # Image B thresholding
        grayB = rgb2gray(imageB)
        grayB = exposure.equalize_adapthist(grayB)
        grayB = cv2.bilateralFilter((grayB * 255).astype(np.uint8), 9, 75, 75)
        mean_intensity = np.mean(grayB)
        std_intensity = np.std(grayB)
        dynamic_threshold = mean_intensity + 5 * std_intensity
        binary_B = (grayB > dynamic_threshold).astype(np.uint8)

        fig, ax = plt.subplots()
        ax.hist(grayB.ravel(), bins=256, range=[0, 255])
        ax.axvline(dynamic_threshold, color='red', linestyle='--')
        hist_path_B = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_B.png")
        fig.savefig(hist_path_B)
        all_output_files.append(hist_path_B)

        overlap = (np.logical_and(cv2.resize(binary_A, (2048, 2048)) > 0, cv2.resize(binary_B, (2048, 2048)) > 0)).astype(np.uint8) * 255
        overlap_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Overlap.png")
        cv2.imwrite(overlap_path, overlap)
        all_output_files.append(overlap_path)

        # Region associations
        region_props = regionprops(label(overlap))
        cell_props = region_props_A
        crystal_to_cell = []
        cell_to_crystals = defaultdict(list)

        for region in region_props:
            region_coords = set(map(tuple, region.coords))
            best_match = None
            max_overlap = 0
            for cell in cell_props:
                cell_coords = set(map(tuple, cell.coords))
                overlap_area = len(region_coords & cell_coords)
                if overlap_area > 0:
                    cell_to_crystals[cell.label].append(region.label)
                if overlap_area > max_overlap:
                    max_overlap = overlap_area
                    best_match = cell.label
            crystal_to_cell.append({
                "Region_Label": region.label,
                "Associated_Cell": best_match,
                "Overlap (pixels)": max_overlap,
                "Region_Area (pixels)": region.area,
                "Region_Area (µm²)": region.area * (PIXEL_TO_UM ** 2)
            })

        df_mapping = pd.DataFrame(crystal_to_cell)
        df_mapping = df_mapping[(df_mapping["Region_Area (µm²)"] < 10) & (df_mapping["Overlap (pixels)"] > 0)]
        df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())
        df_mapping["Total_Cells_with_crystals"] = df_mapping["Associated_Cell"].nunique()
        df_mapping.loc["Total"] = ["", "", "", "Total Area Crystals", df_mapping["Region_Area (µm²)"].sum(), "", ""]

        cell_crystal_df = pd.DataFrame([
            {"Cell_Label": k, "Crystal_Labels": ", ".join(map(str, v)), "Crystal_Count": len(v)}
            for k, v in cell_to_crystals.items()
        ])

        merged_df = df_mapping.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")

        grouped_xlsx_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_All_Datasets.xlsx")
        with pd.ExcelWriter(grouped_xlsx_path, engine="xlsxwriter") as writer:
            region_area_df.to_excel(writer, sheet_name="Cells", index=False)
            df_mapping.to_excel(writer, sheet_name="Crystals", index=False)
            merged_df.to_excel(writer, sheet_name="Cells + Crystals", index=False)
            cell_crystal_df.to_excel(writer, sheet_name="Cell-Crystal Map", index=False)

        # Annotated Image
        annotated_image = cv2.cvtColor(imageA, cv2.COLOR_GRAY2BGR) if imageA.ndim == 2 else imageA.copy()
        for _, mapping in df_mapping.iterrows():
            if pd.notna(mapping["Associated_Cell"]):
                region = next((r for r in region_props if r.label == mapping["Region_Label"]), None)
                if region:
                    min_row, min_col, max_row, max_col = region.bbox
                    cv2.rectangle(annotated_image, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"Cell {int(mapping['Associated_Cell'])}", (min_col, max(min_row - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, lineType=cv2.LINE_AA)

        annotated_image_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Annotated.png")
        cv2.imwrite(annotated_image_path, annotated_image)
        all_output_files.append(annotated_image_path)

        # Save session result
        st.session_state.script2_results.append({
            "bf_name": bf_file.name,
            "excel_path": grouped_xlsx_path,
            "annotated_img_path": annotated_image_path,
            "overlap_path": overlap_path,
            "hist_A_path": hist_path_A,
            "hist_B_path": hist_path_B,
        })

    # Create ZIP
    zip_path_2 = os.path.join(output_dir, "All_Images_histograms.zip")
    with zipfile.ZipFile(zip_path_2, 'w') as zipf_2:
        for file_path in all_output_files:
            zipf_2.write(file_path, arcname=os.path.basename(file_path))
    st.session_state.zip_path_2 = zip_path_2
    st.success("✅ Processing complete!")

# Display Outputs and Download Buttons
if st.session_state.script2_results:
    st.header("📦 Results")

    for result2 in st.session_state.script2_results:
        st.subheader(f"📁 {result2['bf_name']}")
        st.image(result2["annotated_img_path"], caption="Annotated Image")
        st.image(result2["overlap_path"], caption="Overlap Image")

        with open(result2["excel_path"], "rb") as f2:
            st.download_button("📊 Download Dataset", f2, file_name=os.path.basename(result2["excel_path"]),key=f"download_button_{os.path.basename(result2['excel_path'])}")

        #with open(result2["hist_A_path"], "rb") as f2:
        #    st.download_button("📈 Download Histogram A", f2, file_name=os.path.basename(result2["hist_A_path"]))

        #with open(result2["hist_B_path"], "rb") as f2:
        #    st.download_button("📉 Download Histogram B", f2, file_name=os.path.basename(result2["hist_B_path"]))

    with open(st.session_state.zip_path_2, "rb") as zf_2:
        st.download_button("🗂️ Download All Images and Histograms", zf_2, file_name="All_Images_histograms.zip")

