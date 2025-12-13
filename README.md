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
- `report/` – Final project report (PDF)
- `presentation/` – Final presentation slides
- `dataset/` – Dataset access information (external link)

---

## Dataset
The dataset consists of images of hood outer panels labeled as OK or NOK based on mastic stitch quality.

📁 Google Drive (shared):
[https://drive.google.com/drive/folders/1peGi01W_NN6MFJ8kiFkheHIGxofplcdo?usp=sharing]

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
📄 Final Report (PDF):  
[REPORT LINK HERE]

---

## Presentation
📊 Presentation Slides:  
[SLIDES LINK HERE]

🎥 Pre-recorded Presentation Video:  
[PRESENTATION VIDEO LINK HERE]

---

## Demo Video
🎥 Comprehensive Demo Video:  
[DEMO VIDEO LINK HERE]

---

## Tools and Libraries
- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib

---

## Author
Ahmad Abugamea  
ECE 5831 – Neural Networks
