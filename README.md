# Indoor Structure Detector using MiDaS 2

## 📖 Overview
This project is an improved version of the original **Indoor Structure Detector using MiDaS**, designed to achieve more robust and geometrically consistent indoor structure understanding.

The goal of this project is to detect structural elements such as **walls and corners** from monocular depth estimation and further extend the result toward **3D reconstruction and mapping**.

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

#### 🔹 Local (Segmentation-based)
- Apply segmentation on the depth map
- Analyze:
  - segment geometry
  - slope
  - variance
- Extract structural candidates from segmented regions

#### 🔹 Fusion Strategy
- Combine **global orientation cues** with **local segment-level analysis**
- Use a **conservative fusion strategy** to improve robustness

---

### 3. Structure-aware Geometry Reconstruction

Instead of relying directly on segmentation for point cloud generation:

- Extract **depth cross-section**
- Convert to metric space
- Apply **RANSAC line fitting**

#### For WALL:
- Single dominant line estimation

#### For CORNER:
- Two-line RANSAC
- Enforce **Manhattan World assumption (orthogonality)**
- Compute intersection → corner point

👉 This results in a **geometry-consistent point cloud representation**

---

## 🧱 Pipeline




---

## 📊 Results & Observations

### ✔ Strengths
- Stable direction estimation via von Mises modeling
- Improved robustness through global-local fusion
- Geometry-aware reconstruction using RANSAC
- Conservative fusion reduces catastrophic failure cases

---

### ⚠ Limitations

#### 1. Edge-sparse environments
- Global orientation detection becomes unreliable
- Corner detection degrades

#### 2. Sensitivity to lighting conditions
- :contentReference[oaicite:0]{index=0} is affected by brightness and contrast
- Depth quality degradation impacts:
  - segmentation
  - corner detection

#### 3. Segmentation instability
- Highly dependent on depth quality
- Inconsistent segment boundaries in challenging environments

---

## 🔧 Future Work

### 1. Robust Segmentation
- Improve stability under varying lighting conditions
- Introduce more advanced segmentation techniques

### 2. Higher-level Structural Reasoning
- Compare:
  - segmentation patterns
  - edge distributions
- Move toward **higher-order structural inference**

### 3. Learning-based Extension
- Integrate learning-based models for:
  - structure classification
  - geometric consistency validation

### 4. 3D Mapping & Reconstruction
- Extend point cloud output toward:
  - SLAM
  - indoor mapping
  - full 3D reconstruction

---

## 🎯 Conclusion

This project transitions from a **purely heuristic, histogram-based approach** to a more structured pipeline that combines:

- probabilistic orientation modeling (von Mises)
- local geometric reasoning (segmentation)
- robust model fitting (RANSAC)

The result is a more stable and extensible framework for indoor structure understanding, with clear potential for future expansion into full 3D spatial reasoning systems.
