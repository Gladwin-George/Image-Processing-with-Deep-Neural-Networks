# Task 2 – Image Processing with Deep Neural Networks (DNNs)

This repository contains the implementation for **Task 2 of the SCC451 Machine Learning coursework**, focusing on **deep feature extraction, clustering, and classification** using **pretrained DNN models**.

The task demonstrates how frozen deep neural networks can be used as **feature extractors**, producing high-quality latent representations for downstream unsupervised and supervised learning.

---

## 🧠 Model Used
- **DINOv2 (small)** – Vision Transformer (ViT)
- Pretrained, **frozen weights**
- CLS token embeddings used as feature vectors
- Embedding dimension: **384**

No fine-tuning was performed; the model is used purely for representation learning.

---

## 📂 Datasets

### 1️⃣ Oxford-IIIT Pets
- **Classes:** 37 pet breeds
- **Train images:** 3680
- **Test images:** 3669
- **Annotations:** Official train/test split files
- **Task focus:** Multi-class clustering and classification

---

### 2️⃣ Cats vs Dogs
- **Classes:** 2 (Cat, Dog)
- **Images:** 24,958
- **Loading:** Folder-based (ImageFolder)
- **Task focus:** Binary clustering and classification

---

## ⚙️ Workflow Overview

### 1. Image Preprocessing
- Resize images to **224 × 224**
- Convert to tensors
- Normalise using ImageNet statistics
- Ensures compatibility with DINOv2

---

### 2. Feature Extraction
- Batch-wise inference using GPU (if available)
- Extracted **CLS token embeddings**
- Saved features and labels to disk (`.npy`) for reuse
- Avoids recomputation during analysis

---

### 3. Feature Visualisation
Applied dimensionality reduction to inspect latent space structure:
- **PCA (2D)** – global variance structure
- **t-SNE (2D)** – local neighbourhood preservation

Clear class separation is observed, especially for Cats vs Dogs.

---

## 🔍 Clustering

### K-Means Clustering
- Applied after **L2 normalisation**
- PCA used before clustering for stability
- Evaluated using **Davies–Bouldin Index**

**Results:**
- Oxford-IIIT Pets: meaningful multi-class structure
- Cats vs Dogs: clear binary separation

---

## 🧪 Classification (Linear Probe)

### Logistic Regression
Used as a **linear classifier on frozen embeddings**, equivalent to adding a linear head on top of the DNN.

#### Oxford-IIIT Pets
- **Accuracy:** ~94.5%
- Strong diagonal dominance in confusion matrix
- Minor confusion between visually similar breeds

#### Cats vs Dogs
- **Accuracy:** ~99.86%
- Precision, Recall, F1 ≈ 1.0
- Near-perfect class separation

---

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Davies–Bouldin Index
- Confusion Matrix visualisation

---

## 🏆 Key Findings
- DINOv2 embeddings are **highly discriminative**, even without fine-tuning
- Linear classifiers perform exceptionally well on extracted features
- Feature space supports both:
  - Unsupervised clustering
  - Supervised classification
- Performance scales well from binary to fine-grained multi-class problems

---

## 🧪 Requirements
```bash
python >= 3.8
torch
torchvision
transformers
numpy
scikit-learn
matplotlib
seaborn
