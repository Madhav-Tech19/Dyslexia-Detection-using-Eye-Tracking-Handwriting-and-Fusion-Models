# Dyslexia-Detection-using-Eye-Tracking-Handwriting-and-Fusion-Models

This project implements deep learning models for detecting dyslexia using:

Eye-tracking fixation images

Handwriting samples

A late fusion module that combines predictions from both modalities for improved accuracy.

📂 Project Structure
├── eye.py             # Eye-tracking model (MobileNetV2-based, multi-task)
├── handwriting.py     # Handwriting model (LeViT-based classifier)
├── final.py           # Fusion module combining eye-tracking + handwriting
├── requirements.txt   # Python dependencies (to be created)
├── README.md          # Project documentation

⚙️ Requirements

Python 3.8+

PyTorch >= 1.10

Torchvision

timm

scikit-learn

pandas

numpy

matplotlib

seaborn

tqdm

Pillow

Install all dependencies with:

pip install -r requirements.txt

🧩 Models Overview
1. Eye-Tracking Model (eye.py)

Uses MobileNetV2 as a lightweight backbone.

Handles multiple eye-tracking tasks: T1_Syllables, T4_Meaningful_Text, and T5_Pseudo_Text.

Incorporates an attention-based fusion mechanism for combining task features.

Outputs classification (dyslexic / not dyslexic).

Run training:

python eye.py

2. Handwriting Model (handwriting.py)

Uses LeViT (Vision Transformer) as backbone (via timm).

Trains on handwriting images with two classes: normal and reversal.

Saves predictions and trained weights.

Includes detailed training history, confusion matrix, and result saving.

Run training & evaluation:

python handwriting.py

3. Fusion Model (final.py)

Implements late fusion strategy for decision-level combination.

Uses adaptive confidence-weighted fusion.

Evaluates with multiple metrics:

Accuracy, Precision, Recall, F1-score

ROC & PR Curves

Confusion Matrices

Agreement & confidence analysis

Produces a comprehensive evaluation dashboard.

Run fusion:

python final.py

📊 Outputs

Each script generates:

Trained model weights (.pth)

Classification reports (.txt)

Validation metrics (.json, .csv)

Visualizations (.png plots for confusion matrices, ROC, PR curves, training history)

🚀 How to Use

Prepare datasets:

Eye-tracking fixation images in subfolders (Syllables, MeaningfulText, PseudoText).

Handwriting dataset in class folders (normal, reversal).

Ensure corresponding labels CSV is available (dyslexia_class_label.csv).

Train models individually:

python eye.py
python handwriting.py


Generate predictions and save results (.csv).

Run fusion module for combined evaluation:

python final.py

📌 Notes

Designed to run on GPU (CUDA) but will fall back to CPU.

Optimized for low-memory GPUs (e.g., GTX 1050 3GB).

Fusion module can handle missing labels by generating synthetic ground truth.
