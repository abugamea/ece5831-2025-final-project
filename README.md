# ECE 5831 Final Project – OK vs NOK Mastic Stitch Detection

This project presents a lightweight convolutional neural network (CNN) developed for automated visual inspection of mastic stitch quality on automotive hood outer panels. The model performs binary classification (OK vs NOK) and is designed for manufacturing quality inspection applications.

---

## Project Overview
- Task: Binary image classification (OK / NOK)
- Application: Automotive body manufacturing – mastic stitch inspection
- Model: Lightweight CNN (TensorFlow / Keras)
- Evaluation: Accuracy, confusion matrix, classification report

---

## Repository Structure
- `final-project.ipynb` – Main notebook demonstrating training, evaluation, and inference
- `src/` – Python scripts for training and evaluation
- `models/` – Trained model and metadata
- `report/` – Final project report
- `presentation/` – Final presentation slides
- `dataset/` – Dataset access information

---

## Dataset
The dataset consists of images of hood outer panels labeled as OK or NOK based on mastic stitch quality. [Link:](https://drive.google.com/drive/folders/1bB_Vq4jbzmhraG88bzgD4Bj9rJs8LnJG?usp=sharing)

[Google Drive (Complete Folder):](https://drive.google.com/drive/folders/1tbiM2EVUFh9XVvMdo0uaq_989nbHpJWk?usp=sharing)

---

## How to Run
1. Open `final-project.ipynb`
2. Run all cells from top to bottom
3. The notebook will:
   - Load the dataset
   - Load the trained CNN
   - Evaluate performance
   - Display confusion matrix and metrics
   - Run example predictions

---

## Results Summary
- Accuracy: ~98%
- Zero false rejections (OK → NOK)
- Minimal overfitting observed
- Balanced performance across both classes

---

## Project Report
[Final Report (PDF):](https://drive.google.com/file/d/18Iv8lgMyWYv1HufnQZLmKziF_Q5D0zR0/view?usp=sharing)

---

## Presentation
[Presentation Slides:](https://docs.google.com/presentation/d/17Eda1VIVL0zWhk20hiJTZMncmsXaF9kf/edit?usp=sharing&ouid=101540443637033060443&rtpof=true&sd=true)


Pre-recorded Presentation Video:  
[[PRESENTATION VIDEO LINK HERE](https://drive.google.com/file/d/18Iv8lgMyWYv1HufnQZLmKziF_Q5D0zR0/view?usp=sharing)]

---

## Demo Video
Demo Video of Python Code:  
[[DEMO VIDEO LINK HERE](https://youtu.be/HWxMofejkQs)]

---

## Tools and Libraries used in 
- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib

---

## Student
Ahmad Abugamea  
ECE 5831 – Neural Networks and Pattern Recognition
