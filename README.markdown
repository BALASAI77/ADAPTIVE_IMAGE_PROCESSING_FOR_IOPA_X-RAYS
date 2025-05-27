# Adaptive Image Preprocessing Pipeline for IOPA X-rays

## Project Overview

This project develops an adaptive preprocessing pipeline for Intraoral Periapical (IOPA) X-ray images, aimed at enhancing image quality for dental diagnostics, such as caries detection. The pipeline dynamically adjusts preprocessing steps based on image quality metrics (brightness, contrast, sharpness, noise) to improve clarity while preserving critical structural details. This work was developed as part of an internship assignment to advance dental imaging quality, ensuring better visibility of dental features for diagnostic purposes.

### Key Features
- **Adaptive Preprocessing**: Customizes contrast enhancement, noise reduction, and sharpening based on image quality analysis.
- **Quality Metrics**: Evaluates brightness, contrast (RMS), sharpness (Laplacian variance), and noise (wavelet estimate).
- **Performance Metrics**: Measures Structural Similarity Index (SSIM) and edge preservation to balance enhancement with diagnostic detail retention.
- **Visualization**: Generates comparison plots (`output/comparison_<image_id>.png`) showing original, static, and adaptive images alongside pixel intensity histograms, plus interactive plots for deeper analysis.

### Final Results (as of May 27, 2025)
- **Images Processed**: 13 IOPA X-rays
- **Adaptive SSIM**: 0.982 (Static: 0.727, Target: ~0.95)
- **Adaptive Edge Preservation**: 0.9663 (Static: 0.9075, Target: ~0.965)

The adaptive pipeline exceeds static preprocessing by ~35% in SSIM and ~6.5% in edge preservation, achieving the project goals of enhancing IOPA X-ray images while maintaining structural integrity for dental diagnostics.

## Project Structure

```
cavity_processing/
│
├── data/                           # Directory for input IOPA X-ray images (e.g., IS20250115_191316_7227_10120577.dcm, R4.png to R10.png)
├── output/                         # Directory for output visualizations and logs
│   ├── comparison_<image_id>.png   # Comparison plots (original, static, adaptive images and histograms)
│   ├── quality_metrics.json        # Quality metrics for all images
│   ├── summary.json                # Evaluation summary (SSIM, edge preservation)
│   ├── metrics.html                # Distribution of quality metrics
│   ├── ssim.html                   # SSIM comparison across images
│   ├── edge.html                   # Edge preservation comparison across images
│   └── preprocessing.log           # Log file for preprocessing steps and metrics
├── config.py                       # Configuration file for thresholds and visualization settings (version: 49e703dc-a0da-499f-b708-4fea0b03ada3)
├── core.py                         # Core pipeline logic (DICOM handling, preprocessing, evaluation, visualization) (version: 47bb8a19-964a-47e6-a693-7188d3c56b6a)
├── pipeline.py                     # Main script to run the pipeline (version: 9110dcb1-e471-4f91-9894-9f6b87a773126)
├── requirements.txt                # Required Python packages
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation (this file, version: c54bc529-d211-43f0-8ea3-452acecb0b9f)
```

## Installation

### Prerequisites
- **Operating System**: Windows (developed on Windows 10/11)
- **Python**: Version 3.13
- **Dependencies**: Listed in `requirements.txt`

### Setup
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd cavity_processing
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include:
   - `pydicom`: For reading DICOM files
   - `pillow`: For image handling
   - `scikit-image`: For image processing and metrics (SSIM)
   - `pandas`: For data handling
   - `opencv-python`: For image preprocessing
   - `matplotlib`: For visualization
   - `plotly`: For interactive plots

   Verify installation:
   ```bash
   pip show pydicom pillow scikit-image pandas opencv-python matplotlib plotly
   python -c "import pydicom; import PIL; import skimage; import pandas; import cv2; import matplotlib; import plotly"
   ```

3. **Prepare Directories**:
   Ensure the following directories exist:
   ```bash
   mkdir data
   mkdir output
   ```
   - Place your IOPA X-ray images (DICOM or image formats like PNG) in the `data/` directory. Example files: `IS20250115_191316_7227_10120577.dcm`, `R4.png` to `R10.png`.

## Usage

1. **Run the Pipeline**:
   ```bash
   cd "C:/Users/BALA SAI/OneDrive/Desktop/cavity_processing"
   python pipeline.py
   ```
   This script:
   - Loads images from `data/`.
   - Applies static and adaptive preprocessing.
   - Evaluates results using SSIM and edge preservation.
   - Generates visualizations in `output/`.

2. **Check Outputs**:
   - **Visualizations**: `output/comparison_<image_id>.png` shows original, static, and adaptive images with pixel intensity histograms.
   - **Metrics**: `output/summary.json` contains the evaluation summary.
   - **Logs**: `output/preprocessing.log` details preprocessing steps and quality metrics.

## Pipeline Details

### Core Components (`core.py`)
- **DICOMHandler**: Reads DICOM/RVG files and normalizes them to 8-bit grayscale.
- **ImageQualityAnalyzer**: Computes quality metrics (brightness, contrast, sharpness, noise) for adaptive adjustments.
- **StaticPreprocessor**:
  - Applies histogram equalization and Gaussian blur (3x3 kernel).
  - Serves as a baseline for comparison.
- **AdaptivePreprocessor**:
  - Classifies image quality (low/medium/high) based on thresholds in `config.py`.
  - Applies:
    - Contrast enhancement (CLAHE with clipLimit 1.5–3.0, gamma correction for high contrast).
    - Noise reduction (wavelet denoising for high noise, bilateral filter for medium noise).
    - Sharpening (unsharp masking with alpha 1.1–1.5).
  - Includes SSIM feedback: reverts to milder settings if SSIM drops below 0.6 to prevent over-processing.
- **ImageEvaluator**: Computes SSIM and edge preservation (using Canny edge detection) for evaluation.
- **VisualizationManager**:
  - Generates comparison plots with images and histograms.
  - Creates interactive plots (`metrics.html`, `ssim.html`, `edge.html`) for quality metrics distribution and evaluation results.

### Configuration (`config.py`)
- Defines thresholds for quality classification (e.g., brightness 40–200, contrast 25–85).
- Visualization settings (e.g., DPI, show plots).

## Results and Analysis

### Evaluation Metrics
- **SSIM**: Measures structural similarity to the original image. Higher values indicate better preservation of structural details.
- **Edge Preservation**: Measures edge similarity (via Canny edge detection) between original and processed images. Higher values indicate better edge retention.

### Final Performance
- **Adaptive SSIM**: 0.982, a ~35% improvement over static (0.727), exceeding the target of ~0.95.
- **Adaptive Edge Preservation**: 0.9663, a ~6.5% improvement over static (0.9075), meeting the target of ~0.965.
- **Visual Quality**: Adaptive images show enhanced contrast and clarity (e.g., clearer tooth boundaries, cavities) compared to both original and static images, as seen in `output/comparison_<image_id>.png`.

### Visualizations

#### Sample Comparison Plot
Below is a sample comparison plot (`output/comparison_IS20250115_191316_7227_10120577.png`) showing the original, static, and adaptive processed images, along with their pixel intensity histograms. This visualization highlights the adaptive pipeline’s ability to enhance contrast and clarity while preserving structural details.

**How to Include**:
- Upload the image to your GitHub repository (e.g., `output/comparison_IS20250115_191316_7227_10120577.png`) or a hosting service like Google Drive.
- Replace the placeholder URL below with the actual image URL.

![Sample Comparison Plot](https://via.placeholder.com/800x600.png?text=Comparison+Plot+-+Original,+Static,+Adaptive+Images+and+Histograms)
*Caption*: The top row displays the original, static, and adaptive images, with the adaptive image showing enhanced contrast and sharper dental features, such as tooth boundaries and cavities. The bottom row presents pixel intensity histograms: the adaptive histogram (red) demonstrates a balanced intensity spread compared to the original (gray) and static (blue), improving visibility of dental structures without over-flattening.

#### Interactive Quality Metrics Distribution
The `output/metrics.html` file contains an interactive Plotly visualization of quality metrics (brightness, contrast, sharpness, noise) across all 13 images. Below is a text-based representation of what this plot looks like:

```
Quality Metrics Distribution
----------------------------------------
Brightness: [Histogram peaking around 100-150, indicating most images have moderate brightness]
Contrast:   [Histogram showing a spread from 20-80, with most images in the 25-85 ideal range]
Sharpness:  [Histogram with a peak at ~500, showing varied sharpness levels]
Noise:      [Histogram peaking at ~5-10, indicating low noise in most images]
----------------------------------------
```

**How to Include**:
- Host `metrics.html` on a web server or GitHub Pages.
- Link to it in the README:
  ```markdown
  [View Interactive Quality Metrics Distribution](https://your-hosting-url/output/metrics.html)
  ```

#### SSIM and Edge Preservation Comparisons
The `output/ssim.html` and `output/edge.html` files show bar charts comparing SSIM and edge preservation across all images. Here’s a simplified representation:

```
SSIM Comparison
----------------------------------------
Image ID | Static SSIM | Adaptive SSIM
----------------------------------------
R4       | 0.710      | 0.979
R5       | 0.732      | 0.984
...      | ...        | ...
Average  | 0.727      | 0.982
----------------------------------------

Edge Preservation Comparison
----------------------------------------
Image ID | Static Edge | Adaptive Edge
----------------------------------------
R4       | 0.895      | 0.962
R5       | 0.910      | 0.968
...      | ...        | ...
Average  | 0.9075     | 0.9663
----------------------------------------
```

**How to Include**:
- Host `ssim.html` and `edge.html` alongside `metrics.html`.
- Add links in the README:
  ```markdown
  [View SSIM Comparison](https://your-hosting-url/output/ssim.html)  
  [View Edge Preservation Comparison](https://your-hosting-url/output/edge.html)
  ```

## Troubleshooting

- **Low SSIM or Edge Preservation**:
  - Check `preprocessing.log` for applied steps:
    ```bash
    type output\preprocessing.log | findstr "Preprocessing for"
    ```
  - If SSIM < 0.6, the pipeline reverts to milder settings. Adjust SSIM threshold in `core.py` (e.g., 0.65).
  - Fine-tune CLAHE clipLimit (e.g., default to 1.8) or sharpening alpha (e.g., 1.0 for low sharpness).
- **Visual Quality Issues**:
  - If adaptive images are under-enhanced, increase CLAHE clipLimit (e.g., default to 2.5).
  - If over-enhanced, reduce sharpening alpha (e.g., 1.0 for low sharpness).
- **Library Errors**:
  - Reinstall dependencies:
    ```bash
    pip uninstall pydicom pillow scikit-image pandas opencv-python matplotlib plotly -y
    pip install pydicom pillow scikit-image pandas opencv-python matplotlib plotly
    ```

## Future Improvements
- **Dynamic Thresholding**: Adjust `config.py` thresholds dynamically based on image dataset characteristics.
- **Advanced Denoising**: Incorporate non-local means denoising for better edge preservation.
- **Multi-Stage Processing**: Add iterative enhancement with intermediate SSIM/edge checks after each step.
- **Enhanced Visualization**: Add overlays to comparison plots to highlight dental features like cavities more clearly.

## Acknowledgments
- Developed by Bala Sai as part of an internship assignment.
- Built with Python libraries: `pydicom`, `scikit-image`, `opencv-python`, `matplotlib`, `plotly`.

## Contact
For questions or contributions, please reach out to Bala Sai at **balasaimaruboyina0@gmail.com** or via GitHub.

---

*Last updated: May 27, 2025, 08:26 PM IST*
