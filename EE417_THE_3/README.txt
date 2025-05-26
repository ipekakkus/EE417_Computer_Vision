3D Scene Reconstruction via Smartphone Camera Triangulation


This project was developed for EE417 – Computer Vision course at Sabancı University
Submitted by: İpek Akkuş - ipek.akkus@sabanciuniv.edu
Date: 20.05.2025


Project Overview

This project implements a complete stereo vision pipeline to reconstruct a 3D scene from two smartphone images. The pipeline includes image capture, camera calibration, feature detection and matching, fundamental matrix estimation, triangulation, and 3D visualization.


Folder Structure

- THE3_codes.ipynb         # Main Jupyter notebook containing full pipeline
- THE3_report.pdf          # 2–3 page written report with results and discussion
- calibration_data.npz     # Saved camera matrix and distortion coefficients
- dataset/		   # Although there are 4 images taken in this folder, I used: 
    IMG_6470.jpg           # First view of the scene (desk setup)
    IMG_6471.jpg           # Second view of the scene (with ~60% overlap)
- chessboard/
    *.jpg                  # Calibration images of 9x6 checkerboard


Notebook Highlights

- Camera Calibration using checkerboard images and OpenCV
- SIFT and ORB Feature Detection with comparison metrics
- Fundamental Matrix Estimation with epipolar geometry visualization
- Triangulation to recover 3D coordinates from matched keypoints
- 3D Visualization using matplotlib, with intensity coloring and camera poses
- Bounding Box Filtering to remove noisy outliers


Report Summary

- The included THE3_report.pdf contains:
  - Detailed explanation of each pipeline step
  - Results, sample images, 3D point cloud visualizations
  - Comparison of SIFT vs. ORB (accuracy, speed, reprojection error)
  - Error analysis and proposed improvements