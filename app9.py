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
matplotlib.use("Agg")  # Headless-safe backend

from skimage.measure import label, regionprops
from skimage.filters import threshold_li
from skimage.filters import threshold_otsu
from skimage.filters import threshold_isodata
from skimage import data, filters, measure, morphology, exposure
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

import json
from sklearn.cluster import KMeans
from PIL import Image

from pathlib import Path
import shutil

# Streamlit App
st.set_page_config(layout="wide")
st.title("Microscopy Image Processing")

# Initialize rerun flag in session_state if not present
if "rerun_flag" not in st.session_state:
    st.session_state.rerun_flag = False

# File Upload
bf_files = st.file_uploader("Upload BF Images (.tif)", type=["tif"], accept_multiple_files=True)
pl_files = st.file_uploader("Upload PL Images (.tif)", type=["tif"], accept_multiple_files=True)

# Sort uploaded files
if bf_files:
    bf_files = sorted(bf_files, key=lambda x: x.name)
if pl_files:
    pl_files = sorted(pl_files, key=lambda x: x.name)

# File Count Info
if bf_files and pl_files:
    st.success(f"Found {len(bf_files)} BF files and {len(pl_files)} PL files.")
    if len(bf_files) != len(pl_files):
        st.warning("The number of BF and PL images does not match. Only matching pairs will be processed.")

    for bf, pl in zip(bf_files, pl_files):
        st.write(f"Processing: {bf.name} and {pl.name}")

# Output Directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load scale settings
@st.cache_data
def load_scale_settings():
    try:
        with open('scale_map.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"40": 5.64, "100": 13.89}

um_to_px_map = load_scale_settings()

# Sidebar Scale Input
st.sidebar.header("Scale Settings")
selected_um = st.sidebar.selectbox("Known Distance (µm):", list(um_to_px_map.keys()))
distance_in_px = st.sidebar.text_input("Distance in Pixels:", value=str(um_to_px_map.get(selected_um, "")))

# Convert to float with error handling
try:
    s_um = float(selected_um)
    d_px = float(distance_in_px)
    PIXEL_TO_UM = 1 / (s_um / d_px)
    st.success(f"Calibration result: 1 px = {PIXEL_TO_UM:.4f} µm")
    st.session_state.pixel_to_um = PIXEL_TO_UM
except ValueError:
    st.error("Please enter valid numeric values for scale calibration.")

# Add Scale Section
st.sidebar.markdown("---")
st.sidebar.subheader("Manage Scale Settings")

new_um = st.sidebar.text_input("New µm value")
new_px = st.sidebar.text_input("New pixel value")
if st.sidebar.button("➕ Add Scale"):
    try:
        new_um_f = float(new_um)
        new_px_f = float(new_px)
        um_to_px_map[str(int(new_um_f))] = new_px_f
        with open('scale_map.json', 'w') as f:
            json.dump(um_to_px_map, f, indent=4)
        st.sidebar.success(f"Added scale: {int(new_um_f)} µm = {new_px_f} px")
        st.cache_data.clear()
        # Toggle rerun_flag to trigger rerun
        st.session_state.rerun_flag = not st.session_state.rerun_flag
    except ValueError:
        st.sidebar.error("Enter valid numbers to add scale.")

# Delete Scale Option
delete_um = st.sidebar.selectbox("Select µm to delete", list(um_to_px_map.keys()))
if st.sidebar.button("🗑️ Delete Scale"):
    try:
        um_to_px_map.pop(delete_um, None)
        with open('scale_map.json', 'w') as f:
            json.dump(um_to_px_map, f, indent=4)
        st.sidebar.success(f"Deleted scale: {delete_um} µm")
        st.cache_data.clear()
        # Toggle rerun_flag to trigger rerun
        st.session_state.rerun_flag = not st.session_state.rerun_flag
    except Exception as e:
        st.sidebar.error(f"Error deleting: {e}")

# Session State Initialization
if "script1_done" not in st.session_state:
    st.session_state.script1_done = False
if "script1_results" not in st.session_state:
    st.session_state.script1_results = []
if "zip_path_1" not in st.session_state:
    st.session_state.zip_path_1 = None

# Start Button
if st.button("Number of cells with crystals"):
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
        #bf_file.seek(0)
        #pl_file.seek(0)
        
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

        # Apply morphological operations to clean up the binary mask
        binary_A = morphology.opening(binary_A)
        binary_A = morphology.remove_small_objects(binary_A.astype(bool), min_size=500)
        binary_A = morphology.dilation(binary_A, morphology.disk(4))
        binary_A = morphology.remove_small_holes(binary_A, area_threshold=5000)
        binary_A = morphology.closing(binary_A, morphology.disk(4))
        binary_A = (binary_A > 0).astype(np.uint8) * 255

        region_labels_A = label(binary_A)
        region_props_A = regionprops(region_labels_A)

        new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
        label_counter = 1
        #<1500,>=3,=2,=5,=42

        # Compute average threshold based on the mean and standart desviation of region area
        #max_area = max([region.area for region in region_props_A])
        areas = [region.area for region in region_props_A]
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        average = mean_area + std_area

        # Histogram Areas
        fig, ax = plt.subplots()
        ax.hist(areas, bins=20, color='skyblue', edgecolor='black')
        hist_path_Areas = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_Areas.png")
        fig.savefig(hist_path_Areas)
        all_output_files.append(hist_path_Areas)
        
        for region in region_props_A:
            if region.area < average:
                new_label_img[region.slice][region.image] = label_counter
                label_counter += 1
            else:
                coords = np.column_stack(np.where(region.image))
                if len(coords) >= 5:
                    n_clusters = 3
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
        annotated_path = os.path.join(output_dir, f"{bf_file.name}_Segmented_Cells.png")
        cv2.imwrite(annotated_path, overlay_image)
        all_output_files.append(annotated_path)


        region_area_df = pd.DataFrame({
            "Region_Label": [r.label for r in region_props_A],
            "Region_Area (pixels)": [r.area for r in region_props_A],
            "Region_Area (µm²)": [r.area * (PIXEL_TO_UM ** 2) for r in region_props_A]
        })

        region_area_df = region_area_df[region_area_df["Region_Area (µm²)"] > 0]
        total_cells = region_area_df["Region_Label"].count() - 1  # Subtract 1 if you're excluding the bottom-right region
        region_area_df.loc["Total Area"] = ["", "Total Area", region_area_df["Region_Area (µm²)"].sum()]
        region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

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
        dynamic_threshold = mean_intensity + 4 * std_intensity
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
            best_match_cell = None
            max_overlap = 0
            for cell in cell_props:
                cell_coords = set(map(tuple, cell.coords))
                overlap_area = len(region_coords & cell_coords)
                if overlap_area > 0:
                    cell_to_crystals[cell.label].append(region.label)
                if overlap_area > max_overlap:
                    max_overlap = overlap_area
                    best_match_cell = cell.label
            crystal_to_cell.append({
                "Region_Label": region.label,
                "Associated_Cell": best_match_cell,
                "Overlap (pixels)": max_overlap,
                "Region_Area (pixels)": region.area,
                "Region_Area (µm²)": region.area * (PIXEL_TO_UM ** 2)
            })

            # ✅ Store the crystal label for the matched cell
            if best_match_cell is not None:
                cell_to_crystals[best_match_cell].append(region.label)

        df_mapping = pd.DataFrame(crystal_to_cell)
        df_mapping = df_mapping[(df_mapping["Region_Area (µm²)"] < 10) & (df_mapping["Overlap (pixels)"] > 0)]
        df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())
        df_mapping["Total_Cells_with_crystals"] = df_mapping["Associated_Cell"].nunique()
        df_mapping.loc["Total"] = ["", "", "", "Total Area Crystals", df_mapping["Region_Area (µm²)"].sum(), "", ""]

        #cell_crystal_df = pd.DataFrame([
        #    {"Cell_Label": k, "Crystal_Labels": ", ".join(map(str, v)), "Crystal_Count": len(v)}
        #    for k, v in cell_to_crystals.items()
        #])

        # --- Optional: Save cell-to-crystal list (for debugging or export) ---
        cell_crystal_df = pd.DataFrame([
            {
                "Cell_Label": cell_label,
                #"Crystal_Labels": ", ".join(map(str, crystals)),
                #"Crystal_Count": len(crystals)
                "Crystal_Labels": ", ".join(map(str, set(crystals))),  # remove duplicates
                "Crystal_Count": len(set(crystals))                    # correct count
            }
            for cell_label, crystals in cell_to_crystals.items()
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
        st.image(result1["annotated_img_path"], caption="Detections crystals")
        st.image(result1["overlap_path"], caption="Correlation")

        with open(result1["excel_path"], "rb") as f1:
            st.download_button("📊 Download Dataset", f1, file_name=os.path.basename(result1["excel_path"]),mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f"download_button_{os.path.basename(result1['excel_path'])}")

        #with open(result1["hist_A_path"], "rb") as f1:
        #    st.download_button("📈 Download Histogram A", f1, file_name=os.path.basename(result1["hist_A_path"]))

        #with open(result1["hist_B_path"], "rb") as f1:
        #    st.download_button("📉 Download Histogram B", f1, file_name=os.path.basename(result1["hist_B_path"]))

    with open(st.session_state.zip_path_1, "rb") as zf_1:
        #st.download_button("🗂️ Download All Images and Histograms", zf_1, file_name="All_Images_histograms.zip")
        st.download_button(label="📦 Download All Outputs (ZIP)",data=zf_1,file_name=os.path.basename(st.session_state.zip_path_1),mime="application/zip")
#-----------------------------------------------------------------------------------------------------------------------------------

# Session State Initialization
if "script2_done" not in st.session_state:
    st.session_state.script2_done = False
if "script2_results" not in st.session_state:
    st.session_state.script2_results = []
if "zip_path_2" not in st.session_state:
    st.session_state.zip_path_2 = None

# Start Button
if st.button("Areas"):
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

        # Apply morphological operations to clean up the binary mask
        binary_A = morphology.opening(binary_A)
        binary_A = morphology.remove_small_objects(binary_A.astype(bool), min_size=500)
        binary_A = morphology.dilation(binary_A, morphology.disk(4))
        binary_A = morphology.remove_small_holes(binary_A, area_threshold=5000)
        binary_A = morphology.closing(binary_A, morphology.disk(4))
        binary_A = (binary_A > 0).astype(np.uint8) * 255

        region_labels_A = label(binary_A)
        region_props_A = regionprops(region_labels_A)

        new_label_img = np.zeros_like(region_labels_A, dtype=np.int32)
        label_counter = 1
        #<1500,>=3,=2,=5,=42

        # Compute average threshold based on the mean and standart desviation of region area
        #max_area = max([region.area for region in region_props_A])
        areas = [region.area for region in region_props_A]
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        average = mean_area + std_area

        # Histogram Areas
        fig, ax = plt.subplots()
        ax.hist(areas, bins=20, color='skyblue', edgecolor='black')
        hist_path_Areas = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_Areas.png")
        fig.savefig(hist_path_Areas)
        all_output_files.append(hist_path_Areas)
        
        for region in region_props_A:
            if region.area < average:
                new_label_img[region.slice][region.image] = label_counter
                label_counter += 1
            else:
                coords = np.column_stack(np.where(region.image))
                if len(coords) >= 5:
                    n_clusters = 3
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
        annotated_path = os.path.join(output_dir, f"{bf_file.name}_Segmented_Cells.png")
        cv2.imwrite(annotated_path, overlay_image)
        all_output_files.append(annotated_path)

        region_area_df = pd.DataFrame({
            "Region_Label": [r.label for r in region_props_A],
            "Region_Area (pixels)": [r.area for r in region_props_A],
            "Region_Area (µm²)": [r.area * (PIXEL_TO_UM ** 2) for r in region_props_A]
        })

        region_area_df = region_area_df[region_area_df["Region_Area (µm²)"] > 0]
        total_cells = region_area_df["Region_Label"].count() - 1  # Subtract 1 if you're excluding the bottom-right region
        region_area_df.loc["Total Area"] = ["", "Total Area", region_area_df["Region_Area (µm²)"].sum()]
        region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

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
        dynamic_threshold = mean_intensity + 4.6 * std_intensity
        binary_B = (grayB > dynamic_threshold).astype(np.uint8)

        binary_B = opening(binary_B)# Remove small noise
        #binary_B= morphology.dilation(binary_B, morphology.disk(4)) # Dilation
        #binary_B = morphology.closing(binary_B, morphology.disk(4)) # Closing
        binary_B = (binary_B > 0).astype(np.uint8) * 255 # Convert back to binary

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
                    best_match_cell = cell.label
            crystal_to_cell.append({
                "Region_Label": region.label,
                "Associated_Cell": best_match_cell,
                "Overlap (pixels)": max_overlap,
                "Region_Area (pixels)": region.area,
                "Region_Area (µm²)": region.area * (PIXEL_TO_UM ** 2)
            })

            # ✅ Store the crystal label for the matched cell
            if best_match_cell is not None:
                cell_to_crystals[best_match_cell].append(region.label)

        df_mapping = pd.DataFrame(crystal_to_cell)
        df_mapping = df_mapping[(df_mapping["Region_Area (µm²)"] < 6) & (df_mapping["Overlap (pixels)"] > 0)]
        df_mapping["Associated_Cell_Count"] = df_mapping["Associated_Cell"].map(df_mapping["Associated_Cell"].value_counts())
        df_mapping["Total_Cells_with_crystals"] = df_mapping["Associated_Cell"].nunique()
        df_mapping.loc["Total"] = ["", "", "", "Total Area Crystals", df_mapping["Region_Area (µm²)"].sum(), "", ""]

        #cell_crystal_df = pd.DataFrame([
        #    {"Cell_Label": k, "Crystal_Labels": ", ".join(map(str, v)), "Crystal_Count": len(v)}
        #    for k, v in cell_to_crystals.items()
        #])

        # --- Optional: Save cell-to-crystal list (for debugging or export) ---
        cell_crystal_df = pd.DataFrame([
            {
                "Cell_Label": cell_label,
                #"Crystal_Labels": ", ".join(map(str, crystals)),
                #"Crystal_Count": len(crystals)
                "Crystal_Labels": ", ".join(map(str, set(crystals))),  # remove duplicates
                "Crystal_Count": len(set(crystals))                    # correct count
            }
            for cell_label, crystals in cell_to_crystals.items()
        ])
        
        merged_df = df_mapping.merge(region_area_df, left_on="Associated_Cell", right_on="Region_Label", how="inner")

        #-----------------------------------------------------------------------------------------
        # Initialize the column with NaNs
        merged_df["Crystal/Cell Area (%)"] = pd.NA

        # Calculate percentage only for rows except the last two
        merged_df.loc[:-3, "Crystal/Cell Area (%)"] = (
            merged_df.loc[:-3, "Region_Area (µm²)_x"] / merged_df.loc[:-3, "Region_Area (µm²)_y"] * 100
        )
        #-------------------------------------------

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
        st.image(result2["annotated_img_path"], caption="Detection crystals")
        st.image(result2["overlap_path"], caption="Correlation")

        with open(result2["excel_path"], "rb") as f2:
            st.download_button("📊 Download Dataset", f2, file_name=os.path.basename(result2["excel_path"]),key=f"download_button_{os.path.basename(result2['excel_path'])}")

        #with open(result2["hist_A_path"], "rb") as f2:
        #    st.download_button("📈 Download Histogram A", f2, file_name=os.path.basename(result2["hist_A_path"]))

        #with open(result2["hist_B_path"], "rb") as f2:
        #    st.download_button("📉 Download Histogram B", f2, file_name=os.path.basename(result2["hist_B_path"]))

    with open(st.session_state.zip_path_2, "rb") as zf_2:
        st.download_button("🗂️ Download All Images and Histograms", zf_2, file_name="All_Images_histograms.zip")

# Session State Initialization
if "script3_done" not in st.session_state:
    st.session_state.script3_done = False
if "script3_results" not in st.session_state:
    st.session_state.script3_results = []
if "zip_path_3" not in st.session_state:
    st.session_state.zip_path_3 = None

# Start Button
if st.button("Number of cells"):
    if not bf_files or not pl_files:
        st.warning("Please upload both BF and PL files.")
    elif len(bf_files) != len(pl_files):
        st.error("Mismatch in number of BF and PL files.")
    else:
        st.session_state.script3_done = True
        st.session_state.script3_results.clear()

# Processing Logic
if st.session_state.script3_done:
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
        #<1500,>=3,=2,=5,=42

        # Compute average threshold based on the mean and standart desviation of region area
        #max_area = max([region.area for region in region_props_A])
        areas = [region.area for region in region_props_A]
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        average = mean_area + std_area

        # Histogram Areas
        fig, ax = plt.subplots()
        ax.hist(areas, bins=20, color='skyblue', edgecolor='black')
        hist_path_Areas = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_Areas.png")
        fig.savefig(hist_path_Areas)
        all_output_files.append(hist_path_Areas)
        
        for region in region_props_A:
            if region.area < average:
                new_label_img[region.slice][region.image] = label_counter
                label_counter += 1
            else:
                coords = np.column_stack(np.where(region.image))
                if len(coords) >= 5:
                    n_clusters = 3
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
        annotated_path = os.path.join(output_dir, f"{bf_file.name}_Segmented_Annotated.png")
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

        #px_per_um = um_to_px_map[selected_um]  # µm per pixel

        # Create a DataFrame for the regions with their area in µm²
        region_area_df = pd.DataFrame({
            "Region_Label": [region.label for region in region_props_A],
            "Region_Area (pixels)": [region.area for region in region_props_A],
            "Region_Area (µm²)": [r.area * (PIXEL_TO_UM ** 2) for r in region_props_A]
        })
    
        region_area_df = region_area_df[region_area_df["Region_Area (µm²)"] > 0]
        total_cells = region_area_df["Region_Label"].count() - 1  # Subtract 1 if you're excluding the bottom-right region
        region_area_df.loc["Total Area"] = ["", "Total Area", region_area_df["Region_Area (µm²)"].sum()]
        region_area_df.loc["Total Cells"] = ["", "Total Cells", total_cells]

        excel_path = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Region_Area.xlsx")
        region_area_df.to_excel(excel_path, index=False)

        # Histogram A
        fig, ax = plt.subplots()
        ax.hist(grayA.ravel(), bins=256, range=[0, 255])
        ax.axvline(threshold, color='red', linestyle='--')
        hist_path_A = os.path.join(output_dir, f"{os.path.splitext(bf_file.name)[0]}_Histogram_A.png")
        fig.savefig(hist_path_A)
        all_output_files.append(hist_path_A)

        # Save session result
        st.session_state.script3_results.append({
            "bf_name": bf_file.name,
            "annotated_path": annotated_path,
            "hist_A_path": hist_path_A,
            "hist_path_Areas": hist_path_Areas,
            "excel_path": excel_path,
        })

    # Create ZIP
    zip_path_3 = os.path.join(output_dir, "All_Images_histograms.zip")
    with zipfile.ZipFile(zip_path_3, 'w') as zipf_3:
        for file_path in all_output_files:
            zipf_3.write(file_path, arcname=os.path.basename(file_path))
    st.session_state.zip_path_3 = zip_path_3
    st.success("✅ Processing complete!")

# Display Outputs and Download Buttons
if st.session_state.script3_results:
    st.header("📦 Results")

    for result3 in st.session_state.script3_results:
        st.subheader(f"📁 {result3['bf_name']}")
        st.image(result3["annotated_path"], caption="Segmented Image")
        st.image(result3["hist_path_Areas"], caption="Areas Histogram")
        st.image(result3["hist_A_path"], caption="Pixels Intensity Histogram")

        with open(result3["excel_path"], "rb") as f3:
            st.download_button("📊 Download Dataset", f3, file_name=os.path.basename(result3["excel_path"]),key=f"download_button_{os.path.basename(result3['excel_path'])}")

        #with open(result2["hist_A_path"], "rb") as f2:
        #    st.download_button("📈 Download Histogram A", f2, file_name=os.path.basename(result2["hist_A_path"]))

        #with open(result2["hist_B_path"], "rb") as f2:
        #    st.download_button("📉 Download Histogram B", f2, file_name=os.path.basename(result2["hist_B_path"]))

    with open(st.session_state.zip_path_3, "rb") as zf_3:
        st.download_button("🗂️ Download All Images and Histograms", zf_3, file_name="All_Images_histograms.zip")

