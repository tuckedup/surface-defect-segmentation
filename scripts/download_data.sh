#!/bin/bash

# This script provides instructions and URLs for downloading the datasets.
# Some datasets require manual registration or Kaggle API.

echo "=== Surface Defect Segmentation Data Download ==="

# 1. NEU-DET
# URL: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
echo "1. NEU-DET: Download from http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html"
echo "   Extract to data/neu_det/"

# 2. DAGM 2007
# URL: https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-inspection
echo "2. DAGM 2007: Download from https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-inspection"
echo "   Extract to data/dagm/"

# 3. Severstal Steel Defect Detection
# URL: https://www.kaggle.com/c/severstal-steel-defect-detection/data
echo "3. Severstal: Requires Kaggle API: 'kaggle competitions download -c severstal-steel-defect-detection'"
echo "   Extract to data/severstal/"

# 4. MVTec AD
# URL: https://www.mvtec.com/company/research/datasets/mvtec-ad
echo "4. MVTec AD: Download from https://www.mvtec.com/company/research/datasets/mvtec-ad"
echo "   Extract to data/mvtec/"

echo "================================================"
echo "Ensure directory structure matches configs/data.yaml"
