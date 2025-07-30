# 2D Foot Landmark Detection for Robotic Screening

This repository contains a 2D image-processing pipeline for identifying toe landmarks on the soles of human feet using RGB input and OpenCV.

## What It Does

- Converts foot images into binary silhouettes
- Filters valid foot shapes using contour logic
- Identifies five toe tips per foot using geometric segmentation
- Labels toe tip coordinates for downstream robotic guidance or debugging

## Core Logic

The pipeline uses OpenCV and NumPy to:

- Apply Gaussian blur and binary thresholding  
- Filter image contours by area, aspect ratio, and edge exclusion  
- Segment the top foot region and divide it into vertical zones  
- Detect toe peaks from big toe to pinky based on highest points in each slice

## Notes

- This code is fully self-contained and requires only OpenCV, NumPy, and Matplotlib for visualization.
- No machine learning or medical datasets are used — this is purely geometry-based logic.
