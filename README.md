Alzheimer’s Disease Detection using Machine Learning

A machine learning project that detects Alzheimer’s disease from MRI images using deep learning (CNN). This project aims to assist early diagnosis by classifying MRI scans into different Alzheimer’s stages.

📌 Project Overview

Alzheimer’s disease is a progressive neurological disorder that affects memory and cognitive abilities. Early detection can significantly improve treatment outcomes.

This project uses:
Convolutional Neural Networks (CNN)
MRI brain image dataset
Image preprocessing & augmentation
Multi-class classification (Mild, Moderate, Severe, Non-Demented)

🚀 Features

Preprocessing of MRI images
Training a deep learning model (CNN / VGG16 / ResNet — choose your version)
Model evaluation using accuracy, precision, recall, F1-score
Visualizations: loss/accuracy graphs, confusion matrix
Prediction on any new MRI image

🧰 Tech Stack

Python
TensorFlow / Keras
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn
Google Colab / Jupyter Notebook

📦 Installation
git clone https://github.com/Saxshii/alzheimer-s-detection.git
cd alzheimer-s-detection
pip install -r requirements.txt

Run the Model
Train the model:  python src/train.py
Predict on new image: python src/predict.py --image sample.jpg
