# Few‑Shot Face Recognition with Siamese Networks

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains two **Few‑Shot Learning (FSL)** approaches for face recognition using Siamese networks:

- **Contrastive Loss** – [`fsl-contrastive-score.ipynb`](fsl-contrastive-score.ipynb)
- **Triplet Loss** – [`fsl-triplet-score.ipynb`](fsl-triplet-score.ipynb)

Both models are trained on the **CASIA‑WebFace** dataset and evaluated on unseen identities using a **support‑query** protocol, simulating real‑world few‑shot scenarios.

---

## 🧠 Overview

Few‑shot face recognition aims to identify a person from only a few reference images.  
We build two Siamese architectures that learn to map face images into an embedding space where **similar faces are close** and **dissimilar faces are far apart**.

| Approach       | Loss Function          | Training Data        | Evaluation                     |
|----------------|------------------------|----------------------|--------------------------------|
| Contrastive    | `(1-Y)*d² + Y*max(0, m-d)²` | Positive / negative pairs | Support set (5 images/class) + query set (1 image/class) |
| Triplet        | `max(0, d(a,p) - d(a,n) + margin)` | Triplets (anchor, positive, negative) | Same support/query setup |

Both models use a pretrained **FaceNet** (Keras‑Facenet) as the backbone, with additional trainable dense layers on top.

---

## 🔬 Dataset: CASIA‑WebFace

We use a subset of the **CASIA‑WebFace** dataset.  
During training, we **reserve three folders (`000009`, `000032`, `000046`)** for inference – the model never sees these identities during training.  
These held‑out identities are later used to form the **support set** (5 images each) and **query set** (1 image each).

![Sample images from the dataset](images/sample_images.png)  

---

## 📊 Contrastive Learning Notebook

### 1. Data Preparation
- Positive pairs: two images of the **same** identity.
- Negative pairs: two images of **different** identities.  
  ![Example pairs](images/contrastive_pairs.png)  

- The dataset is transformed into a `tf.data.Dataset` with on‑the‑fly preprocessing (resize to 160×160, normalize).

### 2. Model Architecture
- **Backbone**: FaceNet (pretrained on VGGFace2, frozen during initial training).
- **Dense layers** (64 → 32) with dropout and L2 normalization.
- The Siamese network takes two images, passes each through the same feature extractor, and computes the **Euclidean distance** between the embeddings.
- **Loss**: Contrastive loss with margin = 1.0.

```python
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * square_pred + y_true * margin_square)
```
---

## 3. Training
- Optimizer: RMSprop.
- Batch size: 32.
- 5 epochs.

## 4. Inference: Support & Query Sets
- From each held‑out folder, we select:
- Support set: 5 images → used to create a small evaluation dataset (30 pairs after pairing).
- Query set: 1 image → never seen during training.

- Each query image is paired with every support image, and the model outputs similarity scores (Euclidean distance).
The closest support images (lowest distance) are considered the most similar identities.

## support_set
![support_set](images/support_set.png)  

## query_set
![query_set](images/query_set.png)  

Figure 5: The three query images (one per held‑out folder).

---

## 📈 Triplet Learning Notebook
(Similar structure; see the notebook for full details)

- **Triplet generation:** For each anchor, select a positive (same class) and a negative (different class).

- **Triplet loss with margin** = 0.2.
- Training uses the same FaceNet backbone and dense layers.
- Inference identical to the contrastive approach.

---

## 🔍 Final Inference Results

After training, both models are evaluated on the held‑out identities using the support‑query protocol.  
For each query image, we compute its distance to every support image and retrieve the closest matches.  
The examples below show that both models successfully match the query to the correct identity (the support images from the same folder have the smallest distances).

| Model       | Query‑to‑Support matching example |
|-------------|----------------------------------|
| Contrastive | ![Contrastive inference](images/contrastive_inference.png) |
| Triplet     | ![Triplet inference](images/triplet_inference.png) |

*For each query (leftmost column), the three closest support images are shown along with their distance scores. Correct matches (same identity) are highlighted.*

---

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies:
  
```python
pip install tensorflow keras-facenet mtcnn numpy matplotlib opencv-python Pillow
```
3. Download the **CASIA‑WebFace** dataset (or any aligned face dataset) and place it in a folder (e.g., casia dataset/).
4. Open the desired notebook in Jupyter / Kaggle / Google Colab.
5. Adjust the base_dir path to point to your dataset.
6. un all cells.

**Note:** The notebooks were originally developed on Kaggle; you may need to adjust paths and install additional libraries if running locally.

---

## 📌 Results & Observations
- Both models achieve **~82‑83%** validation accuracy after 5 epochs.
- The contrastive model slightly outperforms triplet on this dataset, possibly due to simpler loss and balanced pair sampling.
- The inference setup demonstrates that the network can correctly match query images to the corresponding support images with low distances.

---

## 🛠️ Requirements
- Python 3.7+
- TensorFlow 2.x
- keras‑facenet
- mtcnn
- numpy, matplotlib, opencv‑python, Pillow

---








