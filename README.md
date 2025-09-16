🧠 Multi-Modal Dyslexia Detection with Deep Learning
📖 Research Project Overview

This repository presents the implementation of my proposed research project on dyslexia detection through multi-modal deep learning.

Unlike existing studies that analyze handwriting or eye-tracking independently, this project introduces a novel late fusion framework that integrates both modalities. The approach demonstrates how combining visual-cognitive signals with handwriting patterns can significantly improve the reliability and accuracy of dyslexia screening.

The framework consists of three key components:

Eye-Tracking Analysis – Fixation images across multiple reading tasks are processed using a MobileNetV2-based model with an attention mechanism.

Handwriting Recognition – A LeViT transformer-based classifier captures structural irregularities in handwriting (normal vs reversal).

Adaptive Confidence-Weighted Fusion – A newly designed late fusion module integrates predictions from both modalities for enhanced decision-making.

This research highlights the potential of multi-modal learning in advancing cognitive and educational AI, laying the groundwork for future assistive diagnostic tools.

📂 Project Structure
├── eye.py             # Eye-tracking model (MobileNetV2 with attention)
├── handwriting.py     # Handwriting model (LeViT-based classifier)
├── final.py           # Adaptive late fusion module
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation

⚙️ Requirements

Python 3.8+

PyTorch >= 1.10

Torchvision

timm

scikit-learn

pandas, numpy

matplotlib, seaborn

tqdm

Pillow

Install all dependencies with:

pip install -r requirements.txt

🧩 Research Modules
1. Eye-Tracking Model (eye.py)

Lightweight MobileNetV2 backbone optimized for low-memory GPUs.

Processes multiple reading tasks (Syllables, MeaningfulText, PseudoText).

Employs attention-based fusion across task features.

Outputs dyslexia classification.

Run training:

python eye.py

2. Handwriting Model (handwriting.py)

Uses LeViT (Vision Transformer) for handwriting classification.

Trained on two classes: normal and reversal.

Tracks training history and generates detailed evaluation reports.

Run training & evaluation:

python handwriting.py

3. Fusion Model (final.py)

Implements adaptive confidence-weighted late fusion.

Integrates predictions from both modalities.

Evaluates performance with:

Accuracy, Precision, Recall, F1-score

ROC & Precision-Recall Curves

Confusion Matrices

Model agreement vs correctness

Produces a comprehensive evaluation dashboard.

Run fusion analysis:

python final.py

📊 Research Outputs

Each module generates:

Trained models (.pth)

Evaluation reports (.txt, .json, .csv)

Visualizations (.png) for training curves, confusion matrices, ROC/PR curves, and fusion results

🚀 How to Reproduce

Prepare datasets

Eye-tracking fixation images in subfolders (Syllables, MeaningfulText, PseudoText).

Handwriting dataset in folders (normal, reversal).

Ensure labels CSV (dyslexia_class_label.csv) is available for eye-tracking.

Train each model

python eye.py
python handwriting.py


Generate predictions & results (.csv).

Run fusion for integrated evaluation

python final.py

📌 Research Notes

Designed with GPU compatibility but supports CPU fallback.

Optimized for low-memory GPUs (e.g., GTX 1050 3GB).

Fusion module can automatically handle missing ground truth by generating synthetic labels for evaluation.

📜 License

This project is part of my MSc AI research and is released under the MIT License for academic and research use.
