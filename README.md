# CGMap 🌾🛰️
### A Geospatially Aware Deep Learning Framework for Crop Gap Mapping Using UAV

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/GIS-Rasterio%20%7C%20QGIS-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Conference-CVIP%202025-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Under%20Review-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Institution-Plaksha%20University-red?style=for-the-badge" />
</p>

---

<!-- ====================================================================
     PUBLICATION STATUS BLOCK
     Once the paper is published, comment out the "Under Review" line
     and uncomment the "Published at" line below.
     ==================================================================== -->

> 🕐 **Status:** Paper submitted to the 10th International Conference on Computer Vision and Image Processing (**CVIP 2025**), IIT Ropar, Punjab, India — *currently under review.*

<!-- > ✅ **Published at:** 10th International Conference on Computer Vision and Image Processing (**CVIP 2025**), IIT Ropar, Punjab, India -->

> **Authors:** Karan Sharma, Rajiv Ranjan, Dinesh Kumar, Shashank Tamaskar
> Center for Sustainable & Precision Agriculture, Plaksha University, Mohali, India

---

## 📄 Research Poster

The CVIP 2025 conference poster is available in this repository:

> 📎 **[851_KaranSharma.pdf](./851_KaranSharma.pdf)**

<object data="./851_KaranSharma.pdf" type="application/pdf" width="100%" height="800px">
  <p>Your browser does not support embedded PDFs.
  <a href="./851_KaranSharma.pdf">Download the poster here</a>.</p>
</object>

---

## 🧠 Abstract

Sugarcane germination in India is still assessed through manual inspections — slow, error-prone, and impractical at scale. **CGMap** is a geospatially aware, deep-learning-based pipeline that automatically detects early-stage sugarcane saplings from UAV imagery and identifies missing-plant regions ("gaps").

A lightweight YOLOv8 object detection model provides plant locations, which are converted into a **georeferenced point cloud** for spatial analysis. A **Minimum Spanning Tree (MST)** based orientation normalization technique enables reliable row and column extraction even in irregular Indian farm layouts. Gap detection is performed using expected intra-row spacing criteria, and all outputs are exported in **GIS-ready WKT format** for agronomic decision-making.

CGMap enables timely transplantation interventions, supports resource-efficient field management, and provides a scalable, interpretable, and generalizable solution for **precision agriculture**.

---

## ⚙️ Methodology

The pipeline is an end-to-end geospatial computer-vision system:

```
UAV Imagery
    ↓
Orthomosaic Generation (Asmoli farms, Uttar Pradesh)
    ↓
Farm Boundary Extraction (from GIS polygon coordinates)
    ↓
YOLOv8 Sapling Detection (tiled 300×300px chips)
    ↓
Centroid → Georeferenced Point Cloud
    ↓
MST-Based Field Orientation Normalization
    ↓
Row / Column Structural Extraction (linear fitting + perpendicular thresholding)
    ↓
Gap Detection (inter-plant distance vs. expected spacing)
    ↓
GIS Export (WKT: plant points, row lines, detected gaps)
```

### Key Technical Steps

| Step | Description |
|------|-------------|
| **Data Collection** | UAV surveys over sugarcane farms in the Asmoli region, UP |
| **Orthomosaic Stitching** | High-resolution aerial maps with accurate field surface geometry |
| **Object Detection** | YOLOv8 trained on annotated sapling imagery; confidence threshold: `0.1` |
| **Point Cloud Construction** | Each detection centroid projected to global pixel coordinates |
| **MST Orientation** | Fully connected graph → MST → dominant field orientation vector |
| **Row Extraction** | Point cloud rotated to aligned frame; rows/cols via linear regression |
| **Gap Detection** | Inter-plant distance anomalies flagged as single or consecutive gaps |
| **GIS Export** | WKT-format CSVs: plant points, structural lines, gap annotations |

---

## 🗂️ Repository Structure

```
CGmap/
├── main.py                         # Full pipeline entry point
├── best.pt                         # Trained YOLOv8 model weights
├── detected_plant_points.csv       # Output: detected plant coordinates
├── orthomosiacs/                   # Input orthomosaic .tif files
│   └── model_farm_2.tif
├── data_asmoli_model/              # 📦 Publicly released training dataset (see below)
└── 851_KaranSharma.pdf             # CVIP 2025 conference poster
```

---

## 📦 Dataset — `data_asmoli_model/`

The training data in `data_asmoli_model/` is a **publicly released dataset** made available for the research community.

It was **manually collected via UAV** over sugarcane farms in the **Asmoli region, Uttar Pradesh, India**, and **hand-annotated** by the authors for early-stage sapling detection. This is, to our knowledge, one of the few openly available annotated datasets for sugarcane seedling detection in Indian farm conditions.

<!-- ====================================================================
     DATASET CITATION BLOCK
     Once the paper/dataset DOI is issued, comment out the "coming soon"
     line and uncomment the DOI badge + citation lines below.
     ==================================================================== -->

> 📌 **Dataset DOI / citation coming soon** — will be updated upon publication.

<!-- > [![DOI](https://img.shields.io/badge/DOI-XXXXXXX-blue)](https://doi.org/XXXXXXX) -->
<!-- > If you use this dataset, please cite the accompanying paper (see [Citation](#-citation) below). -->

**You are free to use this dataset for research purposes.** If you do, please credit this repository and the authors until the formal citation is available.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics rasterio opencv-python numpy pandas matplotlib networkx scikit-learn scipy shapely
```

### Configuration

Edit the top of `main.py` to point to your data:

```python
input_file     = 'orthomosiacs/model_farm_2.tif'   # Input orthomosaic
model_path     = 'best.pt'                          # YOLOv8 weights
chip_width     = 300                                # Tile width (px)
chip_height    = 300                                # Tile height (px)
conf_threshold = 0.1                                # Detection confidence
MST_THRESHOLD  = 27                                 # Max MST edge length for row grouping
```

### Run

```bash
python main.py
```

**Outputs:**
- `detected_plant_points.csv` — all detected plant centroids
- `model_farm_1_points.csv` — georeferenced WKT plant points
- `model_farm_5_lines_wkt.csv` — structural row lines in WKT
- Gap annotations ready for QGIS / agronomy dashboard import

---

## 📊 Experimental Results

- **Test Site:** Sugarcane farms, Asmoli region, Uttar Pradesh, India
- **Detection Backbone:** YOLOv8 (model-agnostic — compatible with Faster R-CNN, EfficientDet, etc.)
- **Runtime:** Complete detection + analysis in tens of milliseconds (GPU) / sub-second (mid-range CPU)
- **Edge Deployment:** Practical runtimes on Raspberry Pi 4 and Pi 5 — suitable for field-deployed devices
- **Spatial analysis** scales with number of detected plants, not image resolution → efficient at any scale

---

## 🌍 GIS Integration

All outputs are exported as **WKT (Well-Known Text)** — directly importable into:
- [QGIS](https://qgis.org/) ✅
- ArcGIS ✅
- Any agronomy decision-support system with GIS capability ✅

Gap types detected:

| Symbol | Meaning |
|--------|---------|
| 🟢 Detected plants | Healthy plant present |
| 🔴 Single missing plant | Isolated gap |
| `--` Multiple missing plants | Consecutive bald patch |

---

## 🔮 Future Work

- Expand dataset across **multiple seasons and regions**
- Integrate **multispectral cues** (NDVI, NIR) for improved gap detection
- Add **temporal growth pattern** analysis
- Ground-truth validation against manual emergence surveys
- Generalize framework to other **row crops** with similar planting geometries

---

## 🙏 Acknowledgements

This research was supported by **CNH Industrial** under **Project Pahal**, a CSR initiative. Special thanks to **Keshika Gajbhiye** for manuscript proofreading assistance.

---

## 👤 Author

**Karan Sharma**
M.Eng (Software Systems & Robotics), University of Technology Sydney
Former Research Fellow, Center for Sustainable & Precision Agriculture, Plaksha University

- GitHub: [@karanS08](https://github.com/karanS08)
- Contact: 0802karanS@gmail.com

---

## 📜 Citation

<!-- ====================================================================
     CITATION BLOCK
     Once published, comment out the "pre-publication" note and
     uncomment the BibTeX block below.
     ==================================================================== -->

> 🕐 **Citation will be updated upon publication.** For now, if you use this work or dataset please reference this repository.

<!-- If you use CGMap in your research, please cite: -->

<!--
```bibtex
@inproceedings{sharma2025cgmap,
  title     = {CGMap: A Geospatially Aware Deep Learning Framework for Crop Gap Mapping Using UAV},
  author    = {Sharma, Karan and Ranjan, Rajiv and Kumar, Dinesh and Tamaskar, Shashank},
  booktitle = {10th International Conference on Computer Vision and Image Processing (CVIP 2025)},
  year      = {2025},
  address   = {IIT Ropar, Punjab, India}
}
```
-->

---

<!-- ====================================================================
     FOOTER
     Swap "Submitted to" for "Presented at" once published.
     ==================================================================== -->

<p align="center">Made with 🌾 at <b>Plaksha University</b> | Submitted to <b>CVIP 2025, IIT Ropar</b></p>

<!-- <p align="center">Made with 🌾 at <b>Plaksha University</b> | Presented at <b>CVIP 2025, IIT Ropar</b></p> -->
