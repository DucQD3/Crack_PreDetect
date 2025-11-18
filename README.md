# Crack Pre-Detection Pipeline

A comprehensive PyTorch-based 3D crack detection pipeline for computed tomography (CT) and volumetric imaging applications. 

## Features

### Pipeline Stages
1. **Multi-Scale Hessian Detection** - Detect ridge-like structures using Gaussian derivatives (σ ∈ {0.5, 1.5, 2.5, 3.5, 4.5})
2. **Geometric Feature Extraction** - Extract 4 descriptors per 20³ feature cube:
   - Surface area density (boundary voxel fraction)
   - Max region size (largest connected component)
   - Foreground volume (total crack voxels)
   - Projection anisotropy (13-direction STD)
3. **Projection Anisotropy Analysis** - Distinguish cracks (1D, high STD) from blobs (3D, low STD)
4. **CUSUM Statistical Analysis** - Local change-point detection via 3×3×3 sliding window analysis
5. **P-Value Computation** - Empirical p-values from null distribution (rank-based method)
6. **Adaptive P-Value Modification** - Local FDR control using LAWS procedure
7. **Reconstruction & Upsampling** - Restore original resolution via transposed convolution voting (×20 upsampling)

## Installation

### Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy, Matplotlib, SciPy, tifffile

### Input Format

**Image File** (`input_path`):
- Format: TIFF
- Dimensions: Any 3D array (D, H, W)

**Ground Truth** (`gt_path`):
- Format: TIFF (8-bit uint8)
- Dimensions: Same as input image
- Values: Binary (0 or 1, where 1 = crack/defect)

**Null Distribution** (`null_path`):
- File: background-blank_cusum_norm_nc.npy
- Content: CUSUM statistics from background/non-defect regions

### Customization

Edit parameters in **Section 3 (Parameters)**:

```python
# Hessian scales
sigma_values = [0.5, 1.5, 2.5, 3.5, 4.5]  # Adjust for different crack sizes

# Feature grid resolution
cube_size = 20  # Size of detection cubes (change to 10-30 typically)

# CUSUM window
window_size = 3  # Local analysis window (3-5 typical)

# Adaptive thresholding
tau = 0.1  # Sparsity parameter (0.05-0.5)
alpha = 0.2  # Detection threshold (0.1-0.5, higher = more detections)
```

## Algorithm Details

### Mathematical Foundation

#### Stage 1: Hessian Detection
For each scale σ, compute Hessian matrix:
$$H(x,σ) = \begin{bmatrix} \frac{∂^2I}{∂x^2} & \frac{∂^2I}{∂x∂y} & \frac{∂^2I}{∂x∂z} \\ \vdots & \vdots & \vdots \\ \frac{∂^2I}{∂z^2} \end{bmatrix}$$

Ridge detection: Use max(|H_ij|) to capture high curvature
Thresholding: Binary = 1 if response ≥ μ + 3σ

#### Stage 4: CUSUM Analysis
For local window w at position (i,j,k):
$$\text{CUSUM} = \text{mean}(w) - \text{mean}(complement)$$

Standardized across features for consistent statistics.

#### Stage 5: Adaptive FDR Control
Local sparsity estimate.
Modified p-values: $p^* = p / w$ where $w = π̂/(1-π̂)$


### Evaluation Metrics

Computed metrics (all against downscaled ground truth):

- **Jaccard Index (IoU)**: $|A ∩ B| / |A ∪ B|$ - Intersection over union
- **Precision**: $TP / (TP + FP)$ - False positive control
- **Recall**: $TP / (TP + FN)$ - False negative control


## References

Key papers and methods used:
- Hessian-based ridge detection
- CUSUM change-point analysis
- Local FDR control (LAWS procedure)
- Projection-based anisotropy measures

## Contact

For questions or collaboration inquiries, contact: [nguyentranduc1995@gmail.com]


