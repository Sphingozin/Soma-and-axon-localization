# Soma and Axon Fluorescence Analysis

This script enables interactive detection and quantification of fluorescence intensity in soma and axon regions across concentric rings, from reference and target microscopy images.

## Features

- Load grayscale reference and target images
- Manually define a soma by clicking on its border
- Automatically fit a soma circle and generate concentric rings
- Use a reference image mask (via slider thresholding) to define valid signal regions
- Add axon detection from a third image using another threshold mask
- Calculate mean intensity in each ring for soma and axon masks
- Export the results to an Excel file (`ring_intensities.xlsx`)

## How to Use

1. Run the script:
    ```bash
    python localization_reference_mask_axon_detection.py
    ```

2. Workflow:
    - Select a **reference** and **target** image (TIFF).
    - Use the slider to create a binary mask for the reference image.
    - Manually click the soma edges to fit a soma circle.
    - View the concentric rings drawn around the soma.
    - Select a third image (typically target) to define the axon.
    - Use another threshold slider to mask axon structure.
    - The program calculates ring-wise intensity values for soma and axon regions.
    - Results are saved as `ring_intensities.xlsx` in the reference image's directory.

## Requirements

Install the required packages with:

```bash
pip install numpy pandas matplotlib pillow openpyxl scikit-image
```

## Output

The Excel file includes:
- Ring ID
- Inner and outer radius for each ring
- Reference and target intensities
- Axon intensities for both reference and target
- Intensity ratios

---

Author: Sphingozin  
Email: sphingozin@gmail.com
