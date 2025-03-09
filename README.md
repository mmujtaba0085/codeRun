# codeRun

## Here is the link to the [Demo](https://drive.google.com/file/d/1WvLnaoNE8Fcz_36GQW3MAKlOLmBOdr-f/view?usp=sharing) 

# Diabetic Retinopathy Detection using MobileNetV2

This repository contains the code and model for the **Infyma AI Hackathon 25'** project on **Diabetic Retinopathy Detection**. The goal of this project is to classify retinal images into different stages of diabetic retinopathy using a deep learning model.

## Problem Statement
Diabetic Retinopathy (DR) is a complication of diabetes that affects the eyes and can lead to vision loss if not detected early. The task is to develop an AI model that classifies retinal images into different DR severity levels.

## Dataset
The dataset consists of labeled retinal fundus images, categorized into 5 stages of diabetic retinopathy:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/kushagratandon12/diabetic-retinopathy-balanced/data).

## Model Architecture
The model used in this project is based on **MobileNetV2**, a lightweight and efficient convolutional neural network. The model was fine-tuned on the diabetic retinopathy dataset using **PyTorch**.

## Training
The model was trained for 40 epochs with the following configurations:
- **Optimizer:** AdamW
- **Learning Rate:** 0.001 (with StepLR scheduler)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 32
- **Image Size:** 512x512
- **Data Augmentation:** Random horizontal flip, rotation, and normalization

## Evaluation
The model achieved the following performance on the validation and test sets:
- **Validation Accuracy:** 83.04%
- **Test Accuracy:** 84%



### Classification Report (Test Set)
          precision    recall  f1-score   support

       0       0.72      0.72      0.72      1000
       1       0.78      0.78      0.78       971
       2       0.77      0.75      0.76      1000
       3       0.95      0.97      0.96      1000
       4       0.99      0.99      0.99      1000

accuracy                           0.84      4971



## Requirements
To run the code, you need the following dependencies:
- Python 3.10
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using the `requirements.txt` file.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetic-retinopathy-detection.git
   cd diabetic-retinopathy-detection
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
3. Download the dataset from Kaggle and place it in the input directory.
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook model_training.ipynb
5. The trained model weights are saved as diabetic_retinopathy_.pth.
   
