# 3DGS Evaluation Pipeline

A Python pipeline for evaluating 3D Gaussian Splatting render quality 
using PSNR and SSIM metrics.

## What it does
- Loads rendered and ground truth image pairs
- Computes PSNR and SSIM per view
- Outputs averaged metrics as JSON

## Stack
Python, PyTorch, torchmetrics

## Usage
python pipeline/evaluate.py --renders-dir path/to/renders --gt-dir path/to/gt

## Status
In progress — evaluation pipeline complete, MLflow tracking and API in development.
