# Indoor Structure Detector using MiDaS 2

## 📖 Overview
This project is an improved version of the original **Indoor Structure Detector using MiDaS**, designed to achieve more robust and geometrically consistent indoor structure understanding.

The goal of this project is to detect structural elements such as **walls and corners** from monocular depth estimation and further extend the result toward **3D reconstruction and mapping**.

---

## 🆕 Version 2.1 Update

This project has been further extended to **version 2.1**, introducing a new approach for improving structural inference from depth.

### 🔹 Depth Contour-based Structural Reasoning

The previous segmentation approach relied on grouping pixels based on local similarity, which sometimes led to ambiguity in structural interpretation.

In version 2.1:

- The depth map is **quantized into multiple levels (contour-like regions)**
- Structural cues are extracted from:
  - **distribution of contour regions**
  - **area change patterns**
  - **centroid convergence behavior**

This allows the system to:

- Capture **global geometric tendencies** instead of relying purely on local clustering
- Better distinguish between:
  - **planar structures (walls)** → linear distribution
  - **corner structures** → converging, peak-centered distribution

👉 This significantly reduces ambiguity present in the previous segmentation-based approach.

---

## 🚀 Key Improvements

### 1. Direction Estimation using von Mises Distribution
The previous version relied on histogram-based analysis of:
- edge orientation
- depth gradient

However, this approach was sensitive to noise and sparse edge conditions.

In this improved version:
- **von Mises distribution** is applied to both edge orientations and depth gradients
- Enables **more stable and continuous orientation estimation**
- Handles circular data (angles) more appropriately than standard histograms

---

### 2. Fusion of Global and Local Structure Cues

#### 🔹 Global (Orientation-based)
- Extract dominant directions from:
  - edge orientation
  - depth gradient
- Model distributions using von Mises mixture
- Estimate structural likelihood (e.g., Manhattan alignment)

#### 🔹 Local (Segmentation / Contour-based)
- Apply segmentation and contour-level quantization on the depth map
- Analyze:
  - segment geometry
  - contour distribution
  - centroid convergence
- Extract structural candidates from both local regions and global depth structure

#### 🔹 Fusion Strategy
- Combine **global orientation cues** with **local structural analysis**
- Use a **conservative fusion strategy** to improve robustness

---

### 3. Structure-aware Geometry Reconstruction

Instead of relying directly on segmentation for point cloud generation:

- Extract **depth cross-section**
- Convert to metric space
- Apply **RANSAC line fitting**

#### For WALL:
- Single dominant line estimation  
- Produces **stable and consistent point cloud**

#### For CORNER:
- Detect corner using contour + curvature cues
- Construct L-shape using:
  - dominant direction
  - orthogonal constraint

👉 Aims to produce a **geometry-consistent point cloud representation**

---

## 🧱 Pipeline

<!-- Insert pipeline image here -->
<!-- ![Pipeline](./assets/pipeline.png) -->

---

## 📊 Results & Observations (v2.1)

### ✔ Strengths
- Stable direction estimation via von Mises modeling
- Improved robustness through global-local fusion
- Depth contour reasoning reduces ambiguity
- **Wall detection and reconstruction are highly stable**
- Conservative fusion reduces catastrophic failures

---

### ⚠ Limitations

#### 1. Edge-sparse environments
- Global orientation detection becomes unreliable
- Corner detection degrades

#### 2. Sensitivity to lighting conditions
- Depth estimation quality is affected by brightness and contrast
- Impacts:
  - segmentation
  - contour extraction
  - structural inference

#### 3. Corner Geometry Fitting
- Corner detection has improved,
- but **point cloud reconstruction for corners is still unstable**

Specifically:
- L-shape fitting is not always consistent
- Current methods may:
  - fail under noise
  - produce inaccurate geometry

👉 **Corner detection ≠ reliable corner reconstruction**

---

## 🔧 Future Work

### 1. Robust L-shape Fitting
- Develop a more stable method for:
  - corner geometry reconstruction
  - L-shape fitting aligned with depth structure

### 2. Contour-based Structural Modeling
- Extend contour analysis to:
  - multi-peak structures
  - corridor-like environments

### 3. Adaptive Geometric Constraints
- Introduce constraints based on:
  - spatial consistency
  - indoor structural priors

### 4. Learning-based Extension
- Integrate learning-based models for:
  - structure classification
  - geometric validation

### 5. 3D Mapping & Reconstruction
- Extend toward:
  - SLAM
  - indoor mapping
  - full 3D reconstruction

---

## 🎯 Conclusion

This project evolves from a **histogram-based heuristic system** into a more structured framework combining:

- probabilistic modeling (von Mises)
- contour-based structural reasoning
- geometric model fitting (RANSAC)

With the introduction of **depth contour-based analysis (v2.1)**:

- Structural classification is more stable  
- Wall reconstruction is reliable  

However:

- **Corner reconstruction remains an open challenge**,  
highlighting the need for more advanced geometric fitting strategies.

