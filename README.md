# Personalized Federated Deep Learning for Wireless Link Quality Prediction

## Project

Personalized Federated Deep Learning for Wireless Link Quality Prediction under Non-IID Cell Data.

## Dataset

UJIIndoorLoc Wi-Fi fingerprint dataset.

This implementation performs RSSI-derived wireless link-quality classification using Wi-Fi fingerprint data. It is not a direct 4G/5G cellular dataset.

## Classes

Poor: max RSSI < -75 dBm
Fair: -75 dBm <= max RSSI < -65 dBm
Good: -65 dBm <= max RSSI < -55 dBm
Excellent: max RSSI >= -55 dBm

## Feature Pipeline

Original WAP features: 520
Constant WAP features removed: 55
Usable WAP features: 465
Engineered RSSI features: 7
Total model features: 472

Engineered features:
max_rssi
mean_rssi
median_rssi
std_rssi
detected_aps
top3_mean
top5_mean

UJIIndoorLoc value 100 means no detected signal and is converted to -100 dBm.

## Federated Learning

Client identifier: PHONEID
Number of personalized clients: 16
PHONEIDs: 1, 3, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 24

Each client uses a 70/15/15 train/validation/test split.

## Final Hybrid Deployment

Strategy: margin_hybrid
Margin: 0.25 dBm

## Final Reported Test Results

Locked hybrid test accuracy: 97.5984%
Balanced accuracy: 98.1312%
Weighted precision: 97.6574%
Weighted recall: 97.5984%
Weighted F1: 97.6104%

## Run the UI

Open a terminal inside the app directory.

Install dependencies:
pip install -r requirements.txt

Start:
python app.py

The Gradio interface accepts PHONEID and WAP RSSI measurements.

Example:
WAP001=-55
WAP002=-63
WAP005=-71

## Important

The final personalized Keras models contain Lambda layers. The deployment code rebuilds the exact architecture with explicit Lambda output shapes and loads the trained weights.

No retraining is required to use the included deployed models.

## Project Contents

dataset/ - raw UJIIndoorLoc data
models/ - trained model artifacts
preprocessing/ - scaler, processed federated data and hybrid strategy
results/ - evaluation results
graphs/ - project graphs
app/ - standalone Gradio application
notebooks/ - complete project notebook
source_material/ - extracted/source materials
documentation/ - project metadata and manifest

## GitHub

After extracting this project:
git init
git add .
git commit -m "Complete personalized federated wireless link quality project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main

Do not commit passwords, tokens, API keys, or other private credentials.