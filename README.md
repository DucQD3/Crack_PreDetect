# Crack Pre-Detection Pipeline

A comprehensive PyTorch-based 3D crack detection pipeline for computed tomography (CT) and volumetric imaging applications. This notebook implements a sophisticated 7-stage detection algorithm with multi-scale Hessian analysis, geometric feature extraction, and adaptive statistical hypothesis testing.

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

### Key Capabilities
- **GPU Acceleration** - CUDA support with automatic fallback to CPU
- **Memory Efficient** - Chunked processing for large 3D volumes
- **Comprehensive Visualizations** - 4 publication-quality plots per image
- **Production Ready** - Error handling, progress tracking, metric computation
- **Batch Processing** - Process multiple images with automatic results aggregation

## Installation

### Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy, Matplotlib, SciPy, tifffile

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/crack-detection-pipeline.git
cd crack-detection-pipeline

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scipy scikit-image tifffile
```

## Usage

### Quick Start

```python
# 1. Open the notebook in Jupyter
jupyter notebook Publication_Crack_Detection_Pipeline.ipynb

# 2. Configure paths in Section 2 (Load Image Data)
input_path = "path/to/your/image.tif"
gt_path = "path/to/your/ground_truth.tif"
null_path = "path/to/null_distribution.npy"  # Optional

# 3. Run all cells (Ctrl+A, then Shift+Enter)
# Or run cells sequentially to inspect intermediate results
```

### Input Format

**Image File** (`input_path`):
- Format: TIFF (32-bit float preferred) or any tifffile-compatible format
- Dimensions: Any 3D array (D, H, W)
- Intensity: Preferably normalized to [0, 1] or similar range
- Example: 600×600×600 CT scan

**Ground Truth** (`gt_path`):
- Format: TIFF (8-bit uint8)
- Dimensions: Same as input image
- Values: Binary (0 or 1, where 1 = crack/defect)
- Optional: If not needed for training, can use dummy file

**Null Distribution** (`null_path`):
- Format: NumPy .npy file
- Shape: Any 3D array (typically feature-grid resolution)
- Content: CUSUM statistics from background/non-defect regions
- Optional: Pipeline uses synthetic p-values if file not found

### Output

Results are saved to `results/` directory:
```
results/
├── image_name_detection_result.tif    # Binary detection map (uint8)
└── [Additional outputs from visualizations]
```

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
alpha = 0.2  # Detection threshold (0.1-0.5, lower = more detections)
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

#### Stage 6: Adaptive FDR Control
Local sparsity estimate:
$$\hat{π}(τ) = 1 - \frac{\text{count}(p > τ)}{(1-τ) \cdot n}$$

Modified p-values: $p^* = p / w$ where $w = π̂/(1-π̂)$

### Performance Characteristics

Typical runtime (600×600×600 image):
- GPU (NVIDIA RTX3090): ~1-2 minutes

Memory usage:
- Input image: ~1.4 GB (for 600³ float32)
- Feature maps: ~50-200 MB
- GPU VRAM: ~4-8 GB recommended

### Evaluation Metrics

Computed metrics (all against downscaled ground truth):

- **Dice Coefficient**: $2|A ∩ B| / (|A| + |B|)$ - Overlap measure
- **Jaccard Index (IoU)**: $|A ∩ B| / |A ∪ B|$ - Intersection over union
- **Precision**: $TP / (TP + FP)$ - False positive control
- **Recall**: $TP / (TP + FN)$ - False negative control

## Examples

### Example 1: Single Image Analysis
```python
# Load your 3D image
input_path = "data/sample_ct_scan.tif"
gt_path = "data/sample_ct_scan_label.tif"

# Run entire pipeline in one execution
# See: Publication_Crack_Detection_Pipeline.ipynb
```

### Example 2: Batch Processing
Use the companion notebook: `Publication_Crack_Detection_Pipeline_SemiSynthetic.ipynb`
- Processes 48 images automatically
- Collects all metrics
- Generates summary statistics
- Suitable for semi-synthetic datasets with 48-image structure

## Citation

If you use this pipeline in research, please cite:

```bibtex
@software{crack_detection_2024,
  title={Publication-Quality Crack Detection Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/crack-detection-pipeline}
}
```

## License

[Specify your license - MIT, Apache 2.0, GPL, etc.]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Submit pull request

## Issues & Support

- **Bug Reports**: Open an issue with detailed description and minimal reproducible example
- **Feature Requests**: Open an issue with use case description
- **Questions**: See FAQ section below or open a discussion

## FAQ

### Q: Can I use this for 2D images?
A: This pipeline is designed for 3D volumes. For 2D images, adapt Stage 1 (Hessian) to 2D derivatives and adjust feature extraction accordingly.

### Q: What if I don't have ground truth labels?
A: Pipeline still runs; skip evaluation metrics (Section 12) or comment out comparison code. Useful for exploratory analysis.

### Q: How do I adjust sensitivity?
A: Lower `alpha` (e.g., 0.1) → more detections, higher `alpha` (e.g., 0.3) → fewer detections

### Q: Can I use this on CPU only?
A: Yes! Set `device = "cpu"` or remove CUDA code. Performance will be slower (10-60× depending on image size).

### Q: How do I preprocess my data?
A: Normalize intensity to [0,1] or similar before passing to pipeline. See Stage 2 loading code.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `chunk_depth` (64→32) or use CPU mode |
| Slow performance | Check GPU is being used: `torch.cuda.is_available()` |
| File not found errors | Use absolute paths or adjust working directory |
| Poor detection quality | Adjust `sigma_values` or `alpha` parameters |

## References

Key papers and methods used:
- Hessian-based ridge detection
- CUSUM change-point analysis
- Local FDR control (Benjamini-Hochberg)
- Projection-based anisotropy measures

## Contact

For questions or collaboration inquiries, contact: [your-email@example.com]

---

**Last Updated**: November 2024
**Python Version**: 3.8+
**Status**: Production Ready ✓
