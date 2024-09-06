# Han Nom Text Recognition Streamlit Application

## Overview
This Streamlit application performs Han Nom text recognition using a YOLOv10 detection model and PPOCR system. Users can upload an image, which will be processed to detect and recognize text which are the Han Nom Characters that appear in the picture.

## Requirement
- The environment should have all the packages required in the requirement.txt and YOLOv10 package
`pip install -q -r requirements.txt`
`pip install -q -e ./yolov10/` 
- Needs GPU to run the recognition step

## Features
- **Image Upload:** Allows users to upload images in JPG, JPEG, or PNG formats.
- **Model Inference:** Uses YOLOv10 for object detection on uploaded images.
- **OCR Processing:** Recognizes text in rotated images using an OCR system.

## Run through CLI interface (will return the results for all the images in the input folder):
`python3 main.py`

## We also support a Streamlit version with GUI that lets you choose images from your local PC, use this CLI command:
`streamlit run app_streamlit.py`



