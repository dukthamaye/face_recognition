# Face Recognition using Eigenfaces (PCA)

A face recognition system built from scratch using Principal Component Analysis (PCA) — also known as the **Eigenfaces method**. This project demonstrates how classical machine learning techniques can be applied to identify faces from a database.

## What it does

- Loads a face image dataset (ORL/ATT Faces)
- Preprocesses and flattens images into feature vectors
- Applies PCA to extract the most important facial features (Eigenfaces)
- Trains a classifier to recognize faces
- Evaluates performance using accuracy score, confusion matrix, and classification report

## Tech Stack

- **Python**
- **OpenCV** — image loading and preprocessing
- **NumPy** — matrix operations
- **SciPy** — eigenvalue decomposition for PCA
- **Scikit-learn** — train/test split, evaluation metrics
- **Matplotlib** — visualization of faces and results

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/dukthamaye/facerecognition.git
   cd facerecognition
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python scipy matplotlib numpy scikit-learn
   ```

3. Open the notebook:
   ```bash
   jupyter notebook facerecognition.ipynb
   ```

4. Run all cells — the dataset will be automatically downloaded and extracted.

## Dataset

Uses the **ORL/ATT Face Database** — 40 subjects, 10 images each, at 92×112 pixel resolution in grayscale.

## Results

The model uses Eigenfaces (PCA) to reduce the high-dimensional image data into a lower-dimensional space, then classifies faces based on the nearest neighbor in that space.

## Author

**dukthamaye** — [GitHub](https://github.com/dukthamaye)
